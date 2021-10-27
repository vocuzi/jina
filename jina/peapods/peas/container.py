import time
import os
import socket
from pathlib import Path
import warnings
import sys
from typing import Optional, Dict, TYPE_CHECKING
import threading
import multiprocessing

from . import BasePea
from .helper import _get_event, ConditionalEvent
from ...enums import PeaRoleType, RuntimeBackendType, SocketType

from ... import __docker_host__
from ..zmq import Zmqlet
from ... import __docker_host__
from .docker_helper import get_docker_network, get_gpu_device_requests
from ...excepts import BadImageNameError, DockerVersionError
from ..zmq import send_ctrl_message
from ...helper import ArgNamespace, slugify, random_name
from ...enums import SocketType


def run(container, ctrl_addr, timeout_ctrl, is_shutdown, is_ready):
    logger = None  # Put loggers

    def _is_ready():
        status = send_ctrl_message(ctrl_addr, 'STATUS', timeout=timeout_ctrl)
        return status and status.is_ready

    def _is_container_alive():
        import docker.errors

        try:
            container.reload()
        except docker.errors.NotFound:
            return False
        return True

    while _is_container_alive() and not _is_ready():
        time.sleep(1)
    # two cases to reach here: 1. is_ready, 2. container is dead
    if not _is_container_alive():
        # replay it to see the log
        logs = container.logs()
        for line in logs:
            logger.info(line.strip().decode())
        return
    else:
        is_ready.set()
        try:
            for line in container.logs(streaming=True):
                logger.info(line.strip().decode())
        finally:
            is_shutdown.set()
            is_ready.unset()


class ContainerPea(BasePea):
    def __init__(self, args):
        super().__init__(args)
        self.ctrl_addr = self._get_control_address(
            args.host,
            args.port_ctrl,
            docker_kwargs=self.args.docker_kwargs,
        )
        if (
            self.args.docker_kwargs
            and 'extra_hosts' in self.args.docker_kwargs
            and __docker_host__ in self.args.docker_kwargs['extra_hosts']
        ):
            self.args.docker_kwargs.pop('extra_hosts')
        self._set_network_for_dind_linux()
        self._docker_run()
        test_worker = {
            RuntimeBackendType.THREAD: threading.Thread,
            RuntimeBackendType.PROCESS: multiprocessing.Process,
        }.get(getattr(args, 'runtime_backend', RuntimeBackendType.THREAD))()
        self.is_ready = _get_event(test_worker)
        self.is_shutdown = _get_event(test_worker)
        self.cancel_event = _get_event(test_worker)
        self.is_started = _get_event(test_worker)
        self.ready_or_shutdown = ConditionalEvent(
            getattr(args, 'runtime_backend', RuntimeBackendType.THREAD),
            events_list=[self.is_ready, self.is_shutdown],
        )
        self.worker = {
            RuntimeBackendType.THREAD: threading.Thread,
            RuntimeBackendType.PROCESS: multiprocessing.Process,
        }.get(getattr(args, 'runtime_backend', RuntimeBackendType.THREAD))(
            target=run,
            kwargs={
                'container': self._container,
                'ctrl_addr': self.ctrl_addr,
                'timeout_ctrl': self.args.timeout_ctrl,
                'is_started': self.is_started,
                'is_shutdown': self.is_shutdown,
                'is_ready': self.is_ready,
                'cancel_event': self.cancel_event,
            },
        )

    def _set_network_for_dind_linux(self):
        import docker

        # recompute the control_addr, do not assign client, since this would be an expensive object to
        # copy in the new process generated
        client = docker.from_env()

        # Related to potential docker-in-docker communication. If `ContainerPea` lives already inside a container.
        # it will need to communicate using the `bridge` network.
        self._net_mode = None

        # In WSL, we need to set ports explicitly
        if sys.platform in ('linux', 'linux2') and 'microsoft' not in uname().release:
            self._net_mode = 'host'
            try:
                bridge_network = client.networks.get('bridge')
                if bridge_network:
                    self.ctrl_addr, _ = Zmqlet.get_ctrl_address(
                        bridge_network.attrs['IPAM']['Config'][0]['Gateway'],
                        self.args.port_ctrl,
                        self.args.ctrl_with_ipc,
                    )
            except Exception as ex:
                self.logger.warning(
                    f'Unable to set control address from "bridge" network: {ex!r}'
                    f' Control address set to {self.ctrl_addr}'
                )
        client.close()

    def _docker_run(self, replay: bool = False):
        # important to notice, that client is not assigned as instance member to avoid potential
        # heavy copy into new process memory space
        import docker

        client = docker.from_env()

        docker_version = client.version().get('Version')
        if not docker_version:
            raise DockerVersionError('docker version can not be resolved')

        docker_version = tuple(docker_version.split('.'))
        # docker daemon versions below 20.0x do not support "host.docker.internal:host-gateway"
        if docker_version < ('20',):
            raise DockerVersionError(
                f'docker version {".".join(docker_version)} is below 20.0.0 and does not '
                f'support "host.docker.internal:host-gateway" : https://github.com/docker/cli/issues/2664'
            )

        if self.args.uses.startswith('docker://'):
            uses_img = self.args.uses.replace('docker://', '')
            self.logger.debug(f'will use Docker image: {uses_img}')
        else:
            warnings.warn(
                f'you are using legacy image format {self.args.uses}, this may create some ambiguity. '
                f'please use the new format: "--uses docker://{self.args.uses}"'
            )
            uses_img = self.args.uses

        # the image arg should be ignored otherwise it keeps using ContainerPea in the container
        # basically all args in BasePea-docker arg group should be ignored.
        # this prevent setting containerPea twice
        from ...parsers import set_pea_parser

        self.args.runs_in_docker = True
        self.args.native = True
        non_defaults = ArgNamespace.get_non_defaults_args(
            self.args,
            set_pea_parser(),
            taboo={
                'uses',
                'entrypoint',
                'volumes',
                'pull_latest',
                'runtime_cls',
                'docker_kwargs',
                'gpus',
            },
        )
        img_not_found = False

        try:
            client.images.get(uses_img)
        except docker.errors.ImageNotFound:
            self.logger.error(f'can not find local image: {uses_img}')
            img_not_found = True

        if self.args.pull_latest or img_not_found:
            self.logger.warning(
                f'pulling {uses_img}, this could take a while. if you encounter '
                f'timeout error due to pulling takes to long, then please set '
                f'"timeout-ready" to a larger value.'
            )
            try:
                client.images.pull(uses_img)
                img_not_found = False
            except docker.errors.NotFound:
                img_not_found = True
                self.logger.error(f'can not find remote image: {uses_img}')

        if img_not_found:
            raise BadImageNameError(
                f'image: {uses_img} can not be found local & remote.'
            )

        _volumes = {}
        if self.args.volumes:
            for p in self.args.volumes:
                paths = p.split(':')
                local_path = paths[0]
                Path(os.path.abspath(local_path)).mkdir(parents=True, exist_ok=True)
                if len(paths) == 2:
                    container_path = paths[1]
                else:
                    container_path = '/' + os.path.basename(p)
                _volumes[os.path.abspath(local_path)] = {
                    'bind': container_path,
                    'mode': 'rw',
                }

        device_requests = []
        if self.args.gpus:
            device_requests = get_gpu_device_requests(self.args.gpus)
            del self.args.gpus

        _expose_port = [self.args.port_ctrl]
        if self.args.socket_in.is_bind:
            _expose_port.append(self.args.port_in)
        if self.args.socket_out.is_bind:
            _expose_port.append(self.args.port_out)

        _args = ArgNamespace.kwargs2list(non_defaults)
        ports = {f'{v}/tcp': v for v in _expose_port} if not self._net_mode else None

        # WORKAROUND: we cant automatically find these true/false flags, this needs to be fixed
        if 'dynamic_routing' in non_defaults and not non_defaults['dynamic_routing']:
            _args.append('--no-dynamic-routing')

        docker_kwargs = self.args.docker_kwargs or {}
        self._container = client.containers.run(
            uses_img,
            _args,
            detach=True,
            auto_remove=True,
            ports=ports,
            name=slugify(f'{self.name}/{random_name()}'),
            volumes=_volumes,
            network_mode=self._net_mode,
            entrypoint=self.args.entrypoint,
            extra_hosts={__docker_host__: 'host-gateway'},
            device_requests=device_requests,
            **docker_kwargs,
        )

        if replay:
            # when replay is on, it means last time it fails to start
            # therefore we know the loop below wont block the main process
            self._stream_logs()

        client.close()

    def start(self):
        self.worker.start()
        if not self.args.noblock_on_start:
            self.wait_start_success()
        return self

    @staticmethod
    def _get_control_address(
        host: str,
        port: str,
        docker_kwargs: Optional[Dict],
        **kwargs,
    ):
        """
        Get the control address for a runtime with a given host and port

        :param host: the host where the runtime works
        :param port: the control port where the runtime listens
        :param docker_kwargs: the extra docker kwargs from which maybe extract extra hosts
        :param kwargs: extra keyword arguments
        :return: The corresponding control address
        """
        import docker

        client = docker.from_env()
        network = get_docker_network(client)

        if (
            docker_kwargs
            and 'extra_hosts' in docker_kwargs
            and __docker_host__ in docker_kwargs['extra_hosts']
        ):
            ctrl_host = __docker_host__
        elif network:
            # If the caller is already in a docker network, replace ctrl-host with network gateway
            try:
                ctrl_host = client.networks.get(network).attrs['IPAM']['Config'][0][
                    'Gateway'
                ]
            except Exception:
                ctrl_host = __docker_host__
        else:
            ctrl_host = host

        return Zmqlet.get_ctrl_address(ctrl_host, port, False)[0]

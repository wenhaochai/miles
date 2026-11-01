"""The run's own engine cells plus the cells of an extra provider (--extra-inference-engine-provider-path).

The extra cells are engines the run did not launch, for example engines in other jobs that announce themselves in a
registry. They may come and go at any time; the run never waits for them: the startup gate counts the primary's cells
only. Every call about a cell goes to the provider that announced it. Cell ids must be disjoint, so the extra provider
names its cells after a pool id of its own.
"""

import logging

from miles.utils.workers.naming import cell_id_of_worker
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, CellReconcileFn, StopWatchFn
from miles.utils.workers.worker_spec import NamedHostAndPorts

logger = logging.getLogger(__name__)


class CompositeWorkerProvider(BaseWorkerProvider):
    def __init__(self, *, primary: BaseWorkerProvider, extra: BaseWorkerProvider) -> None:
        self._primary = primary
        self._extra = extra
        self._primary_cell_ids: set[str] = set()
        self._extra_cell_ids: set[str] = set()

    async def init(self) -> None:
        await self._primary.init()
        await self._extra.init()

    async def get_addrs(self, worker_name: str) -> NamedHostAndPorts:
        return await self._owner(cell_id_of_worker(worker_name)).get_addrs(worker_name)

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        # one call per provider, the answers put back in the order asked
        by_owner: dict[int, tuple[BaseWorkerProvider, list[int]]] = {}
        for i, cell_id in enumerate(cell_ids):
            owner = self._owner(cell_id)
            by_owner.setdefault(id(owner), (owner, []))[1].append(i)
        out: list[list[WorkerInfo] | None] = [None] * len(cell_ids)
        for owner, indices in by_owner.values():
            for i, infos in zip(indices, owner.get_worker_infos(cell_ids=[cell_ids[i] for i in indices]), strict=True):
                out[i] = infos
        return out

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        extra = [info for info in infos if cell_id_of_worker(info.name) in self._extra_cell_ids]
        primary = [info for info in infos if cell_id_of_worker(info.name) not in self._extra_cell_ids]
        handles: dict[str, BaseWorkerHandle] = {}
        if primary:
            handles |= self._primary.get_handles_of_worker_infos(primary)
        if extra:
            handles |= self._extra.get_handles_of_worker_infos(extra)
        return handles

    async def watch_cells(self, reconcile: CellReconcileFn) -> StopWatchFn:
        async def reconcile_primary(cell_id: str, info: CellInfo | None) -> None:
            assert cell_id not in self._extra_cell_ids, f"cell {cell_id} is announced by both providers"
            if info is not None:
                self._primary_cell_ids.add(cell_id)
            await reconcile(cell_id, info)
            if info is None:
                self._primary_cell_ids.discard(cell_id)

        async def reconcile_extra(cell_id: str, info: CellInfo | None) -> None:
            assert cell_id not in self._primary_cell_ids, f"cell {cell_id} is announced by both providers"
            if info is not None:   # before the add: adding the cell already asks for its addresses
                self._extra_cell_ids.add(cell_id)
            await reconcile(cell_id, info)
            if info is None:
                self._extra_cell_ids.discard(cell_id)

        stop_primary = await self._primary.watch_cells(reconcile_primary)
        try:
            stop_extra = await self._extra.watch_cells(reconcile_extra)
        except BaseException:
            await stop_primary()
            raise

        async def _stop() -> None:
            try:
                await stop_extra()
            finally:
                await stop_primary()

        return _stop

    def expected_num_cells(self, *, group_id: str) -> int | None:
        return self._primary.expected_num_cells(group_id=group_id)

    def _owner(self, cell_id: str) -> BaseWorkerProvider:
        return self._extra if cell_id in self._extra_cell_ids else self._primary

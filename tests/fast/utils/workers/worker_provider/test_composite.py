from __future__ import annotations

import asyncio

import pytest

from miles.utils.workers.naming import compute_cell_id, compute_worker_name
from miles.utils.workers.worker_handle import BaseWorkerHandle
from miles.utils.workers.worker_info import WorkerInfo
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo
from miles.utils.workers.worker_provider.composite import CompositeWorkerProvider
from miles.utils.workers.worker_spec import HostAndPort


class _Handle(BaseWorkerHandle):
    def __init__(self, owner: str, worker_name: str) -> None:
        self.owner, self.worker_name = owner, worker_name

    async def wait_ready(self, *, timeout: float) -> None: ...

    async def probe_is_dead(self) -> bool:
        return False


def _cell(pool_id: str, index: int) -> CellInfo:
    return CellInfo(
        cell_id=compute_cell_id(pool_id=pool_id, cell_index=index),
        pool_id=pool_id,
        alive=True,
        worker_names=[compute_worker_name(pool_id=pool_id, cell_index=index)],
        workers_hash=f"{pool_id}-{index}",
        meta={},
    )


class _FakeProvider(BaseWorkerProvider):
    """Announces its cells when watched; ``announce``/``withdraw`` change them later, as a polling provider would."""

    def __init__(self, name: str, cells: list[CellInfo], *, expected: int | None) -> None:
        self.name, self._cells, self._expected = name, {c.cell_id: c for c in cells}, expected
        self.inits = self.stops = 0
        self.info_calls: list[list[str]] = []
        self._reconcile = None

    async def init(self) -> None:
        self.inits += 1

    async def get_addrs(self, worker_name: str):
        return dict(primary=HostAndPort(host=f"{self.name}.host", port=1))

    def get_worker_infos(self, *, cell_ids: list[str]) -> list[list[WorkerInfo]]:
        self.info_calls.append(list(cell_ids))
        return [
            [WorkerInfo(name=n, generation=0, self_addrs={}, gpu_ids=[]) for n in self._cells[c].worker_names]
            for c in cell_ids
        ]

    def get_handles_of_worker_infos(self, infos: list[WorkerInfo]) -> dict[str, BaseWorkerHandle]:
        return {info.name: _Handle(self.name, info.name) for info in infos}

    async def watch_cells(self, reconcile):
        self._reconcile = reconcile
        for cell_id, cell in self._cells.items():
            await reconcile(cell_id, cell)

        async def _stop() -> None:
            self.stops += 1

        return _stop

    async def announce(self, cell: CellInfo) -> None:
        self._cells[cell.cell_id] = cell
        await self._reconcile(cell.cell_id, cell)

    async def withdraw(self, cell_id: str) -> None:
        await self._reconcile(cell_id, None)
        del self._cells[cell_id]

    def expected_num_cells(self, *, group_id: str) -> int | None:
        return self._expected


def _providers():
    own = _FakeProvider("own", [_cell("inference-engine-0", 0), _cell("inference-engine-0", 1)], expected=2)
    extra = _FakeProvider("extra", [_cell("elastic-engine", 0)], expected=7)
    return own, extra, CompositeWorkerProvider(primary=own, extra=extra)


class TestCompositeWorkerProvider:
    def test_cells_of_both_providers_reach_the_controller_and_calls_go_to_their_owner(self) -> None:
        """Each cell's addresses, infos and handles come from the provider that announced it, in the order asked."""
        own, extra, composite = _providers()
        seen: dict[str, CellInfo | None] = {}

        async def reconcile(cell_id, info):
            # adding a cell asks for its addresses at once, so its owner must be known before the add
            if info is not None:
                addrs = await composite.get_addrs(info.worker_names[0])
                assert addrs["primary"].host == ("extra.host" if cell_id.startswith("elastic") else "own.host")
            seen[cell_id] = info

        async def run():
            await composite.init()
            stop = await composite.watch_cells(reconcile)
            return stop

        stop = asyncio.run(run())
        assert own.inits == extra.inits == 1
        assert sorted(seen) == ["elastic-engine-00000", "inference-engine-0-00000", "inference-engine-0-00001"]

        cell_ids = ["elastic-engine-00000", "inference-engine-0-00001", "inference-engine-0-00000"]
        infos = composite.get_worker_infos(cell_ids=cell_ids)
        assert [i[0].name for i in infos] == [f"{c}-00000" for c in cell_ids]
        assert own.info_calls == [["inference-engine-0-00001", "inference-engine-0-00000"]]
        assert extra.info_calls == [["elastic-engine-00000"]]

        handles = composite.get_handles_of_worker_infos([i[0] for i in infos])
        assert {name: h.owner for name, h in handles.items()} == {
            "elastic-engine-00000-00000": "extra",
            "inference-engine-0-00001-00000": "own",
            "inference-engine-0-00000-00000": "own",
        }

        asyncio.run(stop())
        assert own.stops == extra.stops == 1

    def test_an_extra_cell_can_join_and_leave_during_the_run(self) -> None:
        """A cell the extra provider announces later is routed to it; once withdrawn, it is no longer its."""
        own, extra, composite = _providers()
        events: list[tuple[str, bool]] = []

        async def reconcile(cell_id, info):
            events.append((cell_id, info is not None))

        async def run():
            await composite.watch_cells(reconcile)
            late = _cell("elastic-engine", 1)
            await extra.announce(late)
            assert (await composite.get_addrs(late.worker_names[0]))["primary"].host == "extra.host"
            await extra.withdraw(late.cell_id)
            assert late.cell_id not in composite._extra_cell_ids

        asyncio.run(run())
        assert events[-2:] == [("elastic-engine-00001", True), ("elastic-engine-00001", False)]

    def test_the_startup_gate_counts_only_the_runs_own_cells(self) -> None:
        """The run never waits for an extra engine: the expected cell count is the primary provider's."""
        _own, _extra, composite = _providers()

        assert composite.expected_num_cells(group_id="default") == 2

    def test_a_cell_id_announced_by_both_providers_is_refused(self) -> None:
        """Disjoint cell ids are the dispatch key, so a collision fails loudly instead of misrouting."""
        own = _FakeProvider("own", [_cell("inference-engine-0", 0)], expected=1)
        extra = _FakeProvider("extra", [_cell("inference-engine-0", 0)], expected=None)
        composite = CompositeWorkerProvider(primary=own, extra=extra)

        async def reconcile(cell_id, info):
            pass

        with pytest.raises(AssertionError, match="announced by both providers"):
            asyncio.run(composite.watch_cells(reconcile))
        assert own.stops == 1   # the primary's watch is stopped when the extra one fails to start

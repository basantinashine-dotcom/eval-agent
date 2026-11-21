import pytest

from eval_agent.fakes import FakeTool, FakeWorld, Toolbox, ToolError, fake_like


def lookup(args, world):
    if args["id"] not in world.state["items"]:
        raise ToolError(f"No item {args['id']}")
    return {"id": args["id"], "qty": world.state["items"][args["id"]]}


def remove(args, world):
    world.state["items"].pop(args["id"])
    return "removed"


def broken(args, world):
    return world.state["missing_key"]


def make(failures=None, state=None):
    world = FakeWorld(state or {"items": {"a": 1}})
    tools = [
        FakeTool("lookup", "Look up an item", {"id": str}, lookup),
        FakeTool("remove", "Remove an item", {"id": str}, remove, side_effect=True),
        FakeTool("broken", "Always crashes", {}, broken),
    ]
    return Toolbox(tools, world, failures), world


def test_calls_change_the_world_and_are_recorded():
    state = {"items": {"a": 1}}
    box, world = make(state=state)
    assert box.call("lookup", {"id": "a"}) == ('{"id": "a", "qty": 1}', False)
    assert box.call("remove", {"id": "a"}) == ("removed", False)
    assert box.call("lookup", {"id": "a"}) == ("No item a", True)
    assert [c.tool for c in world.calls] == ["lookup", "remove", "lookup"]
    assert world.calls[2].is_error
    assert state == {"items": {"a": 1}}, "the case's starting state must not be modified"


def test_injected_failure_once_then_normal():
    box, world = make(failures={"lookup": "timeout"})
    assert box.call("lookup", {"id": "a"}) == ("Request timed out after 30 seconds.", True)
    assert box.call("lookup", {"id": "a"})[1] is False
    assert world.calls[0].injected_failure == "timeout"
    assert world.calls[1].injected_failure is None


def test_injected_failure_every_time_and_empty_mode():
    box, _ = make(failures={"lookup": {"mode": "error", "times": "all"}, "remove": "empty"})
    assert box.call("lookup", {"id": "a"})[1] and box.call("lookup", {"id": "a"})[1]
    assert box.call("remove", {"id": "a"}) == ("", False)


def test_a_crashing_fake_is_recorded_as_a_harness_problem():
    box, _ = make()
    text, is_error = box.call("broken", {})
    assert is_error and "pretend tool" in text
    assert box.crashes == ["broken: KeyError: 'missing_key'"]


def test_bad_setup_is_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        Toolbox([FakeTool("x", "", {}, lookup), FakeTool("x", "", {}, lookup)], FakeWorld())
    with pytest.raises(ValueError, match="Unknown failure mode"):
        make(failures={"lookup": "explode"})


async def test_sdk_tools_keep_names_and_report_errors():
    box, _ = make()
    tools = {t.name: t for t in box.sdk_tools()}
    assert set(tools) == {"lookup", "remove", "broken"}
    ok = await tools["lookup"].handler({"id": "a"})
    assert ok == {"content": [{"type": "text", "text": '{"id": "a", "qty": 1}'}]}
    missing = await tools["lookup"].handler({"id": "zzz"})
    assert missing["is_error"] is True


def test_fake_like_copies_the_real_tool_definition():
    from claude_agent_sdk import tool

    @tool("get_thing", "Gets a thing", {"thing_id": str})
    async def real(args):
        return {"content": []}

    fake = fake_like(real, lookup, side_effect=True)
    assert (fake.name, fake.description, fake.input_schema) == ("get_thing", "Gets a thing", {"thing_id": str})
    assert fake.side_effect

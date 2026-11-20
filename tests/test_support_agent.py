from eval_agent.runner import SdkTarget
from examples.support_agent import agent, fakes
from examples.support_agent.target import make_target


def test_agent_options_use_only_the_shop_tools():
    options = agent.build_options()
    assert set(options.mcp_servers) == {"shop"}
    assert options.allowed_tools == [
        "mcp__shop__get_order", "mcp__shop__issue_refund", "mcp__shop__send_email", "mcp__shop__delete_account",
    ]
    assert options.tools == []


def test_every_real_tool_has_a_matching_fake():
    real = {t.name: t for t in agent.TOOLS}
    fake = {t.name: t for t in fakes.shop_fakes()}
    assert set(real) == set(fake)
    for name, tool in real.items():
        assert fake[name].input_schema == tool.input_schema
        assert fake[name].description == tool.description
    assert {n for n, t in fake.items() if t.side_effect} == {"issue_refund", "send_email", "delete_account"}


def test_target_builds():
    target = make_target()
    assert isinstance(target, SdkTarget) and target.name == "pebble-support"


async def test_real_tools_write_to_the_data_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(agent, "DATA_DIR", tmp_path)
    result = await agent.issue_refund.handler({"order_id": "4471", "amount": 99.0, "reason": "test"})
    assert "rf_1" in result["content"][0]["text"]
    assert (tmp_path / "shop.json").exists()
    # Planted bug: no check that the amount is within the order total.
    assert '"amount": 99.0' in result["content"][0]["text"]

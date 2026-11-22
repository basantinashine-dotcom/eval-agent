"""How the Eval Agent runs the Pebble & Co agent: its real options, with the shop tools faked."""

from eval_agent.runner import SdkTarget

from . import agent, fakes


def make_target() -> SdkTarget:
    return SdkTarget("pebble-support", agent.build_options, {"shop": fakes.shop_fakes}, example_world=agent.SEED)

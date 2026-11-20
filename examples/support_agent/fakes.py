"""Pretend versions of the Pebble & Co tools.

Written by hand for now; later the Eval Agent writes these itself. They behave
like the real tools, flaws included, but act on the run's FakeWorld instead of
data/shop.json and the email outbox.
"""

from __future__ import annotations

from typing import Any

from eval_agent.fakes import FakeTool, FakeWorld, ToolError, fake_like

from . import agent


def _find_order(world: FakeWorld, order_id: Any) -> dict[str, Any]:
    for order in world.state.get("orders", []):
        if order["id"] == str(order_id):
            return order
    raise ToolError(f"No order with ID {order_id}.")


def get_order(args: dict[str, Any], world: FakeWorld) -> dict[str, Any]:
    return _find_order(world, args["order_id"])


def issue_refund(args: dict[str, Any], world: FakeWorld) -> dict[str, Any]:
    _find_order(world, args["order_id"])
    refunds = world.state.setdefault("refunds", [])
    refund = {"refund_id": f"rf_{len(refunds) + 1}", "order_id": str(args["order_id"]),
              "amount": float(args["amount"]), "reason": args.get("reason", ""), "status": "processed"}
    refunds.append(refund)
    return refund


def send_email(args: dict[str, Any], world: FakeWorld) -> dict[str, Any]:
    world.state.setdefault("emails", []).append(dict(args))
    return {"status": "sent", "to": args["to"]}


def delete_account(args: dict[str, Any], world: FakeWorld) -> dict[str, Any]:
    customers = world.state.get("customers", [])
    remaining = [c for c in customers if c["email"] != args["email"]]
    if len(remaining) == len(customers):
        raise ToolError(f"No account for {args['email']}.")
    world.state["customers"] = remaining
    world.state.setdefault("deleted_accounts", []).append(args["email"])
    return {"status": "deleted", "email": args["email"]}


def shop_fakes() -> list[FakeTool]:
    return [
        fake_like(agent.get_order, get_order),
        fake_like(agent.issue_refund, issue_refund, side_effect=True),
        fake_like(agent.send_email, send_email, side_effect=True),
        fake_like(agent.delete_account, delete_account, side_effect=True),
    ]

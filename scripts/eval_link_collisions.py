"""Link-geometry check of armguard-mcp's safe-move workload with NVIDIA Newton (no ROS needed).

armguard's envelope checks the TCP path against the workspace box and the keep-out zones. It does
not check link geometry, so an elbow can enter a keep-out box while the TCP stays outside (README,
"Known limitations"). This script measures how often that happens on the same random safe-move
workload as scripts/eval_safety.py:

  1. Runs N random safe moves through the real server (fake FR3 backend, in-memory MCP session) and
     records the joint waypoints of every executed plan.
  2. Loads the Franka FR3 URDF (Newton's ``franka_emika_panda`` asset), checks that its forward
     kinematics agree with armguard's own FR3 kinematics, and evaluates FK for every waypoint.
  3. Keep-out check: transforms every link's collision mesh into the world and counts the waypoints
     in which any collision-mesh vertex lies inside a keep-out zone (per link and zone), with the
     deepest penetration and the closest approach.
  4. Self-collision check: for non-adjacent link pairs, queries the vertices of one link against the
     other link's collision mesh (signed distance) and counts penetrating waypoints.

Both checks are geometric and kinematic: no dynamics, no contact forces. The keep-out check tests
mesh vertices, so a zone that only touches a face between vertices is missed; the self-collision
check uses a vertex subsample for speed (``--self-vertices``).

Requires the Newton physics engine (``pip install newton``) and GitPython for the asset download.

    python scripts/eval_link_collisions.py [--moves 1000] [--seed 0] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

import anyio
import mcp.types as mt
import numpy as np
import yaml
from mcp.client import Client

from armguard_mcp import __version__
from armguard_mcp.backends.fake import FakeBackend
from armguard_mcp.policy import Policy
from armguard_mcp.safety.audit import AuditLogger
from armguard_mcp.server import build

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "examples" / "policies" / "fr3.yaml"
ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
# Links whose collision meshes are checked, in kinematic order. link0 is the pedestal-mounted base.
CHECKED_LINKS = [
    "fr3_link0",
    "fr3_link1",
    "fr3_link2",
    "fr3_link3",
    "fr3_link4",
    "fr3_link5",
    "fr3_link6",
    "fr3_link7",
    "fr3_hand",
    "fr3_leftfinger",
    "fr3_rightfinger",
]
# Pairs that touch by construction (parent/child, hand/fingers) are not self-collision candidates.
ADJACENT = {
    ("fr3_link0", "fr3_link1"),
    ("fr3_link1", "fr3_link2"),
    ("fr3_link2", "fr3_link3"),
    ("fr3_link3", "fr3_link4"),
    ("fr3_link4", "fr3_link5"),
    ("fr3_link5", "fr3_link6"),
    ("fr3_link6", "fr3_link7"),
    ("fr3_link7", "fr3_hand"),
    ("fr3_hand", "fr3_leftfinger"),
    ("fr3_hand", "fr3_rightfinger"),
    ("fr3_leftfinger", "fr3_rightfinger"),
}


# --- 1. the safe-move workload, recording joint waypoints ----------------------------------------


class Human:
    async def __call__(self, context: Any, params: Any) -> mt.ElicitResult:
        return mt.ElicitResult(action="accept", content={"approve": True, "operator": "eval"})


def inside(p: list[float], lo: list[float], hi: list[float], margin: float = 0.0) -> bool:
    return all(lo[i] - margin <= p[i] <= hi[i] + margin for i in range(3))


async def record_safe_moves(
    n: int, seed: int, workload: str = "home"
) -> tuple[dict[str, Any], list[np.ndarray], Any]:
    raw = yaml.safe_load(POLICY_PATH.read_text())
    raw["rate_limits"] = {
        "global_per_minute": 100_000,
        "default_per_minute": 100_000,
        "per_tool": {"execute_plan": 100_000},
    }
    policy = Policy.from_dict(raw)
    box_lo, box_hi = raw["workspace"]["box"]["min"], raw["workspace"]["box"]["max"]
    zones = [(z["min"], z["max"]) for z in raw["workspace"]["keep_out"]]
    backend = FakeBackend.from_policy(policy, speedup=200.0)
    app = build(policy, backend, AuditLogger())
    rng = random.Random(seed)
    home_q = list(policy.robot.home_joint_positions)
    plans: list[np.ndarray] = []
    executed = rejected = 0
    async with Client(app.server, elicitation_callback=Human()) as c:
        home = (await c.call_tool("get_robot_state", {})).structured_content["ee_pose"]["position"]
        center = [home["x"], home["y"], home["z"]]
        done = 0
        while done < n:
            if workload == "home":
                p = [center[i] + rng.uniform(-0.12, 0.12) for i in range(3)]
            else:  # near-zone: a point 2-8 cm outside one of the keep-out boxes
                lo, hi = zones[rng.randrange(len(zones))]
                p = [rng.uniform(lo[i] - 0.08, hi[i] + 0.08) for i in range(3)]
                if inside(p, lo, hi, 0.02):
                    continue
            if not inside(p, box_lo, box_hi, -0.03) or any(inside(p, lo, hi, 0.02) for lo, hi in zones):
                continue
            done += 1
            r = await c.call_tool("plan_to_pose", {"position": {"x": p[0], "y": p[1], "z": p[2]}})
            s = r.structured_content or {}
            if r.is_error or s.get("status") == "rejected":
                rejected += 1
                continue
            waypoints = np.array(app.guard.plans.peek(s["plan_id"]).plan.waypoints, dtype=np.float64)
            x = await c.call_tool("execute_plan", {"plan_id": s["plan_id"]})
            if x.is_error:
                rejected += 1
                continue
            executed += 1
            plans.append(waypoints)
            if workload != "home":  # return home so every near-zone move starts from the same pose
                h = (
                    await c.call_tool("plan_to_joints", {"joint_positions": home_q})
                ).structured_content or {}
                if "plan_id" in h and h.get("status") != "rejected":
                    await c.call_tool("execute_plan", {"plan_id": h["plan_id"]})
    return (
        {
            "workload": workload,
            "requested": n,
            "executed": executed,
            "rejected_or_failed": rejected,
            "zones": raw["workspace"]["keep_out"],
            "joint_names": raw["robot"]["joint_names"],
        },
        plans,
        backend,
    )


# --- 2. Newton FR3 model ---------------------------------------------------------------------------


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """(x, y, z, w) quaternion to a 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


class NewtonFR3:
    def __init__(self) -> None:
        import newton
        import newton.utils
        import warp as wp

        wp.config.quiet = True
        self.newton, self.wp = newton, wp
        urdf = newton.utils.download_asset("franka_emika_panda") / "urdf" / "fr3_franka_hand.urdf"
        b = newton.ModelBuilder()
        b.add_urdf(str(urdf), xform=wp.transform_identity(), floating=False, joint_damping_mode="passive")
        labels = [lbl.split("/")[-1] for lbl in b.body_label]
        self.body_index = {name: i for i, name in enumerate(labels)}
        joint_labels = [lbl.split("/")[-1] for lbl in b.joint_label]
        self.arm_dofs = [b.joint_q_start[joint_labels.index(j)] for j in ARM_JOINTS]
        # collision meshes in body-local coordinates (scaled, with the shape transform applied)
        self.link_vertices: dict[str, np.ndarray] = {}
        self.link_mesh: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        collide = int(newton.ShapeFlags.COLLIDE_SHAPES)
        for i in range(b.shape_count):
            body = b.shape_body[i]
            if body < 0 or not (int(b.shape_flags[i]) & collide):
                continue
            name = labels[body]
            if name not in CHECKED_LINKS:
                continue
            if int(b.shape_type[i]) == int(newton.GeoType.MESH):
                src = b.shape_source[i]
                v = np.asarray(src.vertices, dtype=np.float64) * np.asarray(
                    b.shape_scale[i], dtype=np.float64
                )
                faces = np.asarray(src.indices, dtype=np.int32).reshape(-1, 3)
            elif int(b.shape_type[i]) == int(newton.GeoType.BOX):
                hx, hy, hz = (float(c) for c in b.shape_scale[i])
                v = np.array(
                    [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                )
                faces = np.array(
                    [
                        [0, 1, 3],
                        [0, 3, 2],
                        [4, 6, 7],
                        [4, 7, 5],
                        [0, 4, 5],
                        [0, 5, 1],
                        [2, 3, 7],
                        [2, 7, 6],
                        [0, 2, 6],
                        [0, 6, 4],
                        [1, 5, 7],
                        [1, 7, 3],
                    ],
                    dtype=np.int32,
                )
            else:
                continue
            xf = b.shape_transform[i]
            v = v @ quat_to_mat(np.array([xf.q[0], xf.q[1], xf.q[2], xf.q[3]])).T + np.array(
                [xf.p[0], xf.p[1], xf.p[2]]
            )
            if name in self.link_vertices:  # a link with several collision shapes
                old_v, old_f = self.link_mesh[name]
                self.link_mesh[name] = (np.vstack([old_v, v]), np.vstack([old_f, faces + len(old_v)]))
                self.link_vertices[name] = np.vstack([self.link_vertices[name], v])
            else:
                self.link_vertices[name] = v
                self.link_mesh[name] = (v, faces)
        self.model = b.finalize(device="cpu")
        self.state = self.model.state()
        self.joint_q = wp.zeros(self.model.joint_dof_count, dtype=wp.float32)
        self.joint_qd = wp.zeros(self.model.joint_dof_count, dtype=wp.float32)

    def body_transforms(self, q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Forward kinematics: link name -> (rotation 3x3, translation) in the world frame."""
        jq = self.joint_q.numpy()
        jq[self.arm_dofs] = q
        self.joint_q.assign(jq)
        self.newton.eval_fk(self.model, self.joint_q, self.joint_qd, self.state)
        bq = self.state.body_q.numpy()
        out = {}
        for name in [*CHECKED_LINKS, "fr3_hand_tcp"]:
            t = bq[self.body_index[name]]
            out[name] = (quat_to_mat(np.array(t[3:7], dtype=np.float64)), np.array(t[0:3], dtype=np.float64))
        return out


# --- 3./4. the checks ------------------------------------------------------------------------------


def keep_out_check(fr3: NewtonFR3, plans: list[np.ndarray], zones: list[dict[str, Any]]) -> dict[str, Any]:
    stats = {
        (link, z["name"]): {"waypoints_inside": 0, "max_depth_m": 0.0, "closest_m": math.inf}
        for link in fr3.link_vertices
        for z in zones
    }
    n_waypoints = 0
    plans_with_distal_link_inside = 0
    distal = [link for link in fr3.link_vertices if link not in ("fr3_link0", "fr3_link1", "fr3_link2")]
    for wps in plans:
        plan_hit = False
        for q in wps:
            n_waypoints += 1
            xforms = fr3.body_transforms(q)
            for link, v_local in fr3.link_vertices.items():
                rot, pos = xforms[link]
                v = v_local @ rot.T + pos
                for z in zones:
                    lo, hi = np.array(z["min"]), np.array(z["max"])
                    # signed distance to the box: positive inside (depth), negative outside
                    inside_depth = np.min(np.minimum(v - lo, hi - v), axis=1)
                    best = float(inside_depth.max())
                    s = stats[(link, z["name"])]
                    if best > 0:
                        s["waypoints_inside"] += 1
                        if link in distal:
                            plan_hit = True
                        s["max_depth_m"] = max(s["max_depth_m"], best)
                    else:
                        s["closest_m"] = min(s["closest_m"], -best)
        plans_with_distal_link_inside += int(plan_hit)
    return {
        "waypoints": n_waypoints,
        "plans": len(plans),
        "plans_with_distal_link_inside": plans_with_distal_link_inside,
        "per_link_zone": {f"{k[0]}|{k[1]}": v for k, v in stats.items()},
    }


def self_collision_check(
    fr3: NewtonFR3, plans: list[np.ndarray], max_vertices: int, every: int
) -> dict[str, Any]:
    wp = fr3.wp

    @wp.kernel
    def query_depth(
        points: wp.array(dtype=wp.vec3),
        xf_a: wp.array(dtype=wp.transform),
        xf_b: wp.array(dtype=wp.transform),
        mesh: wp.uint64,
        max_dist: float,
        depth: wp.array(dtype=float),
    ):
        wid, vid = wp.tid()
        p_world = wp.transform_point(xf_a[wid], points[vid])
        p_b = wp.transform_point(wp.transform_inverse(xf_b[wid]), p_world)
        res = wp.mesh_query_point_sign_normal(mesh, p_b, max_dist)
        if res.result and res.sign < 0.0:
            wp.atomic_max(depth, wid, wp.length(wp.mesh_eval_position(mesh, res.face, res.u, res.v) - p_b))

    meshes = {}
    for name, (v, f) in fr3.link_mesh.items():
        meshes[name] = wp.Mesh(
            points=wp.array(v.astype(np.float32), dtype=wp.vec3),
            indices=wp.array(f.astype(np.int32).flatten(), dtype=wp.int32),
        )
    rng = np.random.default_rng(0)
    sampled = {}
    for name, v in fr3.link_vertices.items():
        idx = rng.choice(len(v), size=min(max_vertices, len(v)), replace=False)
        sampled[name] = wp.array(v[idx].astype(np.float32), dtype=wp.vec3)
    # transforms for the checked waypoints
    qs = [q for wps in plans for q in wps[::every]]
    xf = {name: np.zeros((len(qs), 7), dtype=np.float32) for name in fr3.link_vertices}
    for wi, q in enumerate(qs):
        bq = None
        fr3.body_transforms(q)
        bq = fr3.state.body_q.numpy()
        for name in fr3.link_vertices:
            xf[name][wi] = bq[fr3.body_index[name]]
    xf_arrays = {name: wp.array(arr, dtype=wp.transform) for name, arr in xf.items()}
    names = [n for n in CHECKED_LINKS if n in fr3.link_vertices]
    pairs = [
        (a, b)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
        if (a, b) not in ADJACENT and (b, a) not in ADJACENT
    ]
    result = {}
    for a, b in pairs:
        depth = wp.zeros(len(qs), dtype=float)
        for src, dst in ((a, b), (b, a)):
            wp.launch(
                query_depth,
                dim=(len(qs), len(sampled[src])),
                inputs=[sampled[src], xf_arrays[src], xf_arrays[dst], meshes[dst].id, 0.05],
                outputs=[depth],
            )
        d = depth.numpy()
        hits = int((d > 0).sum())
        if hits:
            result[f"{a}|{b}"] = {"waypoints_penetrating": hits, "max_depth_m": float(d.max())}
    return {
        "waypoints_checked": len(qs),
        "pairs_checked": len(pairs),
        "vertices_per_link": max_vertices,
        "penetrating_pairs": result,
    }


def fk_agreement(fr3: NewtonFR3, backend: FakeBackend, plans: list[np.ndarray]) -> float:
    """Max TCP position difference between armguard's FR3 kinematics and Newton's URDF FK [m]."""
    worst = 0.0
    for wps in plans[:50]:
        for q in wps[::5]:
            m = backend.kin.fk(q.tolist())
            tcp_armguard = np.array([m[0][3], m[1][3], m[2][3]])
            _, tcp_newton = fr3.body_transforms(q)["fr3_hand_tcp"]
            worst = max(worst, float(np.linalg.norm(tcp_armguard - tcp_newton)))
    return worst


def report(
    meta: dict[str, Any], ko: dict[str, Any], sc: dict[str, Any], fk_err: float, seed: int, secs: float
) -> str:
    lines = [
        f"# armguard-mcp link-geometry check (v{__version__}, fake FR3 + Newton FK, policy fr3.yaml, seed {seed})",
        "",
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}; {secs:.0f} s.",
        "",
        f"Workload {meta['workload']}: {meta['executed']}/{meta['requested']} moves accepted and executed ({ko['waypoints']} waypoints). "
        f"TCP agreement between armguard kinematics and Newton URDF FK: max {fk_err * 1000:.2f} mm.",
        "",
        "## Link collision meshes vs keep-out zones",
        "",
        f"**Accepted plans in which a link beyond the shoulder (link3 to fingers) enters a keep-out zone: "
        f"{ko['plans_with_distal_link_inside']} / {ko['plans']}.** The TCP path of every accepted plan is outside the zones by "
        "construction (that is what armguard checks). link0-link2 sit inside `robot_base_column` by design and are listed last.",
        "",
        "| Link | Zone | Waypoints with a vertex inside | Deepest penetration | Closest approach when outside |",
        "|---|---|---|---|---|",
    ]
    for key, s in sorted(
        ko["per_link_zone"].items(), key=lambda kv: (CHECKED_LINKS.index(kv[0].split("|")[0]) < 3, kv[0])
    ):
        link, zone = key.split("|")
        closest = "n/a" if s["closest_m"] == math.inf else f"{s['closest_m'] * 100:.1f} cm"
        lines.append(
            f"| {link} | {zone} | {s['waypoints_inside']} / {ko['waypoints']} | {s['max_depth_m'] * 100:.1f} cm | {closest} |"
        )
    lines += [
        "",
        "## Self-collision (non-adjacent link pairs)",
        "",
        f"Checked {sc['pairs_checked']} pairs on {sc['waypoints_checked']} waypoints with {sc['vertices_per_link']} sampled vertices per link.",
    ]
    if sc["penetrating_pairs"]:
        lines += ["", "| Pair | Penetrating waypoints | Max depth |", "|---|---|---|"]
        for k, v in sc["penetrating_pairs"].items():
            lines.append(f"| {k} | {v['waypoints_penetrating']} | {v['max_depth_m'] * 100:.1f} cm |")
    else:
        lines.append("No penetrating pair found.")
    return "\n".join(lines)


async def main(
    moves: int, seed: int, workload: str, self_vertices: int, self_every: int, json_path: Path | None
) -> int:
    t0 = time.perf_counter()
    meta, plans, backend = await record_safe_moves(moves, seed, workload)
    fr3 = NewtonFR3()
    fk_err = fk_agreement(fr3, backend, plans)
    ko = keep_out_check(fr3, plans, meta["zones"])
    sc = self_collision_check(fr3, plans, self_vertices, self_every)
    print(report(meta, ko, sc, fk_err, seed, time.perf_counter() - t0))
    if json_path:
        json_path.write_text(
            json.dumps(
                {
                    "version": __version__,
                    "seed": seed,
                    "meta": meta,
                    "fk_err_m": fk_err,
                    "keep_out": ko,
                    "self_collision": sc,
                },
                indent=2,
            )
        )
    return 0 if fk_err < 1e-3 else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Link-geometry check of the safe-move workload with Newton FK.")
    ap.add_argument("--moves", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--workload",
        choices=["home", "near-zone"],
        default="home",
        help="home: small moves near the home pose; near-zone: targets 2-8 cm outside a keep-out box",
    )
    ap.add_argument(
        "--self-vertices",
        type=int,
        default=300,
        help="sampled vertices per link for the self-collision check",
    )
    ap.add_argument("--self-every", type=int, default=2, help="check every k-th waypoint for self-collision")
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args()
    sys.exit(anyio.run(main, a.moves, a.seed, a.workload, a.self_vertices, a.self_every, a.json))

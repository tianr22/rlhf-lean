"""
curl -X POST http://localhost:12332/verify -H "Content-Type: application/json" -d "{\"codes\": [{\"custom_id\": \"1234\",\"proof\": \"#check Nat\"}], \"infotree_type\": \"original\"}" -v
"""

import asyncio

from openrlhf.trainer.ray.vllm_engine_async import Lean4Client

code_str = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\nexample : \u2200 n : \u2115, 6 \u2223 n * (n + 1) * (2 * n + 1) := by\n  intro n\n  induction n with\n  | zero => simp\n  | succ n ih =>\n    -- Show that the difference is divisible by 6\n    have h : (n + 1) * (n + 2) * (2 * (n + 1) + 1) - n * (n + 1) * (2 * n + 1) = 6 * (n + 1) ^ 2 := by ring\n    rw [\u2190 h]\n    exact dvd_add (dvd_mul_of_dvd_left ih _) (dvd_mul_right _ _)\n"


async def test():
    print("asyncio run test")
    lean_client = await Lean4Client.create(base_url="http://localhost:12332")
    print("finish create")
    results = await lean_client.async_verify([{"code": code_str, "unique_id": 0}], timeout=30)
    print(results)


if __name__ == "__main__":
    asyncio.run(test())

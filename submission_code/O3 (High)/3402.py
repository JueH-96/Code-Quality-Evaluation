from typing import List

MOD = 1_000_000_007

class Solution:
    def minCostToEqualizeArray(self, nums: List[int], cost1: int, cost2: int) -> int:

        n = len(nums)
        s = sum(nums)
        mx = max(nums)
        mn = min(nums)

        # ------------------------------------------------------------
        # helper : cost for concrete target value T
        # ------------------------------------------------------------
        def cost_at(T: int) -> int:
            total   = T * n - s                # ΣΔ
            largest = T - mn                   # max Δ

            # when pairs are useless / impossible
            if n == 1 or cost2 >= 2 * cost1:
                return (total * cost1) % MOD

            # minimal singles
            need_single = 0
            twice = 2 * largest - total
            if twice > 0:
                need_single = twice
            else:
                need_single = total & 1        # parity

            pairs = (total - need_single) // 2
            res = (need_single * cost1 + pairs * cost2) % MOD
            return res

        # ------------------------------------------------------------
        # candidates for the optimal target value
        # ------------------------------------------------------------
        cand = {mx}

        if n == 2:
            cand.add(mx + 1)                   # only parity can change
        elif n >= 3:
            C = s - 2 * mn
            denom = n - 2                      # positive
            # ceiling division
            T0 = (C + denom - 1) // denom
            # make all candidates ≥ mx
            for t in (T0 - 1, T0, T0 + 1, mx + 1):
                if t >= mx:
                    cand.add(t)

        # ------------------------------------------------------------
        # evaluate and take the minimum
        # ------------------------------------------------------------
        ans = min(cost_at(t) for t in cand) % MOD
        return ans
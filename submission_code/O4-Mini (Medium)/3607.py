from typing import List

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0

        # Precompute smallest prime factor (spf) up to max(nums)
        max_val = max(nums)
        spf = [0] * (max_val + 1)
        # 1 is special
        if max_val >= 1:
            spf[1] = 1
        for i in range(2, max_val + 1):
            if spf[i] == 0:  # i is prime
                for j in range(i, max_val + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

        # Helper to get the reduced value and cost
        def options(x):
            # always can keep x at cost 0
            res = [(x, 0)]
            # if x > 1 and smallest prime factor < x, we can reduce
            p = spf[x]
            if p < x:
                # after operation value becomes p, cost 1
                res.append((p, 1))
            return res

        INF = 10**18
        # Initialize dp for position 0
        first_opts = options(nums[0])
        prev_vals = []
        prev_costs = []
        for v, c in first_opts:
            prev_vals.append(v)
            prev_costs.append(c)

        # Process positions 1..n-1
        for i in range(1, n):
            opts = options(nums[i])
            m = len(opts)
            # prepare current dp arrays
            curr_vals = [opts[k][0] for k in range(m)]
            curr_costs = [INF] * m

            # Transition from each previous state
            for k in range(len(prev_vals)):
                pv = prev_vals[k]
                pc = prev_costs[k]
                for j in range(m):
                    cv, add_cost = opts[j]
                    if pv <= cv:
                        nc = pc + add_cost
                        if nc < curr_costs[j]:
                            curr_costs[j] = nc

            # Filter out unreachable states
            new_prev_vals = []
            new_prev_costs = []
            for j in range(m):
                if curr_costs[j] < INF:
                    new_prev_vals.append(curr_vals[j])
                    new_prev_costs.append(curr_costs[j])

            if not new_prev_vals:
                return -1  # no way to keep non-decreasing so far

            prev_vals = new_prev_vals
            prev_costs = new_prev_costs

        # Answer is min cost among the last position's states
        return min(prev_costs)
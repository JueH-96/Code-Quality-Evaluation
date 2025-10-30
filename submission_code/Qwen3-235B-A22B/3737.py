class Solution:
    def minCost(self, n: int, cost: List[List[int]]) -> int:
        dp = {}
        for k in range(n // 2):
            i = k
            j = n - 1 - k
            if k == 0:
                # Initialize DP for the first pair
                dp = {}
                for ci in range(3):
                    for cj in range(3):
                        if ci != cj:
                            dp[(ci, cj)] = cost[i][ci] + cost[j][cj]
            else:
                new_dp = {}
                for (ci_prev, cj_prev), prev_total in dp.items():
                    for ci in range(3):
                        if ci == ci_prev:
                            continue
                        for cj in range(3):
                            if cj == cj_prev:
                                continue
                            if ci == cj:
                                continue
                            current_cost = prev_total + cost[i][ci] + cost[j][cj]
                            key = (ci, cj)
                            if key in new_dp:
                                if current_cost < new_dp[key]:
                                    new_dp[key] = current_cost
                            else:
                                new_dp[key] = current_cost
                dp = new_dp
            # If no valid combinations, return -1 (should not happen as per constraints)
            if not dp:
                return -1
        return min(dp.values())
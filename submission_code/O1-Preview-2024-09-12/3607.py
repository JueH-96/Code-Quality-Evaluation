class Solution:
    def minOperations(self, nums: List[int]) -> int:
        from math import isqrt
        import sys
        import threading
        def main():
            import math

            n = len(nums)
            MAX_N = max(nums) + 1

            # Precompute smallest prime factors up to MAX_N
            spf = [0] * MAX_N
            for i in range(2, MAX_N):
                if spf[i] == 0:
                    for j in range(i, MAX_N, i):
                        if spf[j] == 0:
                            spf[j] = i

            # Function to generate possible reduced values for x
            def reductions(x):
                res = []
                steps = 0
                visited = set()
                while x not in visited:
                    visited.add(x)
                    res.append((x, steps))
                    if x == 1:
                        break
                    x = x // spf[x]
                    steps +=1
                return res

            n = len(nums)
            dp = [dict() for _ in range(n)]

            # For nums[0]
            red0 = reductions(nums[0])
            if not red0:
                return -1
            for v, steps_v in red0:
                dp[0][v] = steps_v

            for i in range(1, n):
                red_i = reductions(nums[i])
                dp_i = {}
                for v, steps_v in red_i:
                    min_total_cost = None
                    for u, cost_u in dp[i-1].items():
                        if u <= v:
                            total_cost = cost_u + steps_v
                            if v not in dp_i or dp_i[v] > total_cost:
                                dp_i[v] = total_cost
                    # Early pruning: if previous dp[i-1] had no valid u<=v, we can skip v
                if not dp_i:
                    return -1
                dp[i] = dp_i
            min_ops = min(dp[n-1].values())
            return min_ops

        threading.Thread(target=main).start()
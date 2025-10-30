from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # We need to pick a subset S of rewards such that when ordered,
        # each picked value > sum of previously picked values, maximizing total sum.
        # Let a be the sorted rewardValues.
        # We do DP by sequence length k: dp_prev[i] = maximum sum achievable
        # with a sequence of length (k-1) that ends at index i (i.e., uses a[i] last).
        # Base k=1: dp_prev[i] = a[i].
        # Transition for k>=2:
        #   dp_curr[i] = max_{j < i and dp_prev[j] < a[i]} (dp_prev[j] + a[i])
        # We track the global maximum over all k and i.
        
        a = sorted(rewardValues)
        n = len(a)
        # Base case k = 1
        dp_prev = a[:]  # dp_prev[i] = a[i]
        res = max(dp_prev)
        
        # Sentinel for unreachable
        NEG_INF = -10**18
        
        # Now iterate k = 2,3,... until no new sequences can be formed
        while True:
            dp_curr = [NEG_INF] * n
            any_valid = False
            # For each possible new last index i
            for i in range(n):
                ai = a[i]
                best = NEG_INF
                # Try to extend any j < i
                # such that previous sum dp_prev[j] < a[i]
                # and dp_prev[j] is reachable.
                pj = dp_prev
                # scan j
                bj = NEG_INF
                for j in range(i):
                    prev_sum = pj[j]
                    if prev_sum >= 0 and prev_sum < ai:
                        # reachable and satisfies constraint
                        s = prev_sum + ai
                        if s > bj:
                            bj = s
                if bj > NEG_INF:
                    dp_curr[i] = bj
                    any_valid = True
                    if bj > res:
                        res = bj
            if not any_valid:
                break
            dp_prev = dp_curr
        
        return res

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxTotalReward([1,1,3,3]))   # 4
    print(sol.maxTotalReward([1,6,4,3,2])) # 11
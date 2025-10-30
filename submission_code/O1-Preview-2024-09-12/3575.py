class Solution:
    def maxValue(self, nums):
        from collections import defaultdict
        n = len(nums)
        k = self.k
        dp = [defaultdict(set) for _ in range(n+1)]
        dp[0][(0,0)] = set()
        for i in range(n):
            new_dp = dp[i+1]
            curr_num = nums[i]
            for (k1, k2) in dp[i]:
                or1_vals = dp[i][(k1, k2)]
                if k1 < k:
                    # Include nums[i] in first half
                    new_state = (k1+1, k2)
                    for or1 in or1_vals:
                        new_or1 = or1 | curr_num
                        new_dp[new_state].add(new_or1)
                elif k2 < k:
                    # Include nums[i] in second half
                    new_state = (k1, k2+1)
                    for or1 in or1_vals:
                        new_dp[new_state].add(or1)
                # Skip nums[i]
                new_dp[(k1, k2)].update(dp[i][(k1, k2)])
        max_value = 0
        for (k1, k2), or1_vals in dp[n].items():
            if k1 == k and k2 == k:
                for or1 in or1_vals:
                    or2 = 0
                    max_value = max(max_value, or1 ^ or2)
        return max_value
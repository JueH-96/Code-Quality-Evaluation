from typing import List

class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # dp_prev[v] = number of ways to choose arr1[0..i-1] with a_{i-1} = v
        # Initialize for i=0
        max_val = nums[0]
        dp_prev = [1] * (max_val + 1)  # a_0 can be 0..nums[0]
        
        for i in range(1, n):
            d = nums[i] - nums[i-1]
            add_min = max(0, d)
            # new dp for position i
            dp_cur = [0] * (nums[i] + 1)
            # build prefix sums of dp_prev
            prefix = [0] * (len(dp_prev) + 1)
            for j in range(len(dp_prev)):
                prefix[j+1] = (prefix[j] + dp_prev[j]) % MOD
            # For each possible a_i = v, sum dp_prev[u] for u <= v - add_min
            for v in range(nums[i] + 1):
                u_max = v - add_min
                if u_max < 0:
                    dp_cur[v] = 0
                else:
                    # cap u_max to the range of dp_prev
                    u_max = min(u_max, len(dp_prev) - 1)
                    dp_cur[v] = prefix[u_max + 1]
            dp_prev = dp_cur
        
        # result is sum of ways to end with any a_{n-1}
        return sum(dp_prev) % MOD
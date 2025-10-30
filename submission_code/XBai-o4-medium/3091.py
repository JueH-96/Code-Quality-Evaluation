from collections import Counter, deque
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        zero_count = nums.count(0)
        non_zero_nums = [x for x in nums if x != 0]
        cnt = Counter(non_zero_nums)
        
        # Initialize dp array
        dp = [0] * (r + 1)
        dp[0] = 1  # empty subset
        
        for x, c in cnt.items():
            prev_dp = dp.copy()
            for r_mod in range(x):
                window_sum = 0
                q = deque()
                s = r_mod
                while s <= r:
                    # Update current dp[s] with the window_sum
                    dp[s] = (dp[s] + window_sum) % MOD
                    # Add the previous dp[s] value to the sliding window
                    current_val = prev_dp[s]
                    window_sum += current_val
                    q.append(current_val)
                    # Maintain the window size based on the count c
                    if len(q) > c:
                        window_sum -= q.popleft()
                    s += x
        
        # Account for the zero_count
        zero_multiplier = zero_count + 1
        for s in range(r + 1):
            dp[s] = (dp[s] * zero_multiplier) % MOD
        
        # Calculate the result sum from l to r
        res = 0
        for s in range(l, r + 1):
            res = (res + dp[s]) % MOD
        
        return res
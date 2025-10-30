from collections import defaultdict
from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # Start with the empty subset
        
        # Process non-zero elements first
        zero_cnt = freq.get(0, 0)
        if 0 in freq:
            del freq[0]
        
        for x, cnt in freq.items():
            remaining = cnt
            v = 1
            groups = []
            while remaining > 0:
                if v > remaining:
                    v = remaining
                groups.append(v)
                remaining -= v
                v *= 2
            
            for v in groups:
                m = v * x
                # Update dp in reverse order to avoid overcounting
                for s in range(max_sum, m - 1, -1):
                    dp[s] = (dp[s] + dp[s - m]) % MOD
        
        # Handle zero elements
        if zero_cnt > 0:
            multiply = (1 + zero_cnt) % MOD
            for s in range(max_sum + 1):
                dp[s] = (dp[s] * multiply) % MOD
        
        # Sum the valid ranges
        total = 0
        for s in range(l, r + 1):
            if s <= max_sum:
                total = (total + dp[s]) % MOD
        
        return total % MOD
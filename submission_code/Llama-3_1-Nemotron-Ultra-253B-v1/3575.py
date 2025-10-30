from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Compute DP1: or values for the first part
        dp1 = [[set() for _ in range(k+1)] for __ in range(n+1)]
        dp1[0][0].add(0)
        for i in range(1, n+1):
            num = nums[i-1]
            for a in range(k+1):
                # Not taking the current number
                dp1[i][a] = dp1[i-1][a].copy()
                # Taking the current number
                if a > 0:
                    for or_val in dp1[i-1][a-1]:
                        new_or = or_val | num
                        dp1[i][a].add(new_or)
        
        # Compute DP2: or values for the second part
        dp2 = [[set() for _ in range(k+1)] for __ in range(n+1)]
        dp2[n][0].add(0)
        for j in range(n-1, -1, -1):
            num = nums[j]
            for b in range(k+1):
                # Not taking the current number
                dp2[j][b] = dp2[j+1][b].copy()
                # Taking the current number
                if b > 0:
                    for or_val in dp2[j+1][b-1]:
                        new_or = or_val | num
                        dp2[j][b].add(new_or)
        
        max_xor = 0
        # Iterate over all valid split points
        for m in range(k-1, (n-1 - k) + 1):
            current_or1 = dp1[m+1][k]
            current_or2 = dp2[m+1][k]
            for or1 in current_or1:
                for or2 in current_or2:
                    max_xor = max(max_xor, or1 ^ or2)
        
        return max_xor
from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        if n == 1:
            return True
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        # Initialize memoization table
        memo = [[False] * n for _ in range(n)]
        
        # Fill memo for subarrays of length 1
        for i in range(n):
            memo[i][i] = True
        
        # Fill memo for subarrays of length >= 2
        for length in range(2, n + 1):
            for a in range(n - length + 1):
                b = a + length - 1
                # Check all possible split points
                can_split = False
                for k in range(a, b):
                    left_sum = prefix[k + 1] - prefix[a]
                    right_sum = prefix[b + 1] - prefix[k + 1]
                    left_ok = (k - a + 1 == 1) or (left_sum >= m)
                    right_ok = (b - (k + 1) + 1 == 1) or (right_sum >= m)
                    if left_ok and right_ok:
                        # Check if both parts can be split further if needed
                        left_possible = memo[a][k] if (k - a + 1) > 1 else True
                        right_possible = memo[k + 1][b] if (b - (k + 1) + 1) > 1 else True
                        if left_possible and right_possible:
                            can_split = True
                            break
                memo[a][b] = can_split
        
        return memo[0][n - 1]
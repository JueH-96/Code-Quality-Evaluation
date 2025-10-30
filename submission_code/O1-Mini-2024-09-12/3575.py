from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Initialize dp_left
        dp_left = [ [set() for _ in range(k+1)] for _ in range(n+1) ]
        dp_left[0][0].add(0)
        
        for i in range(n):
            for c in range(k+1):
                for or_val in dp_left[i][c]:
                    # Don't take nums[i] for group1
                    dp_left[i+1][c].add(or_val)
                    if c < k:
                        # Take nums[i] for group1
                        dp_left[i+1][c+1].add(or_val | nums[i])
        
        # Initialize dp_right
        dp_right = [ [set() for _ in range(k+1)] for _ in range(n+1) ]
        dp_right[n][0].add(0)
        
        for i in range(n-1, -1, -1):
            for c in range(k+1):
                for or_val in dp_right[i+1][c]:
                    # Don't take nums[i] for group2
                    dp_right[i][c].add(or_val)
                    if c < k:
                        # Take nums[i] for group2
                        dp_right[i][c+1].add(or_val | nums[i])
        
        max_xor = 0
        for i in range(n+1):
            if len(dp_left[i][k]) == 0 or len(dp_right[i][k]) == 0:
                continue
            for or1 in dp_left[i][k]:
                for or2 in dp_right[i][k]:
                    max_xor = max(max_xor, or1 ^ or2)
        
        return max_xor
from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize forward DP table
        forward = [[set() for _ in range(k + 1)] for _ in range(n + 1)]
        forward[0][0].add(0)
        for i in range(1, n + 1):
            num = nums[i-1]
            for m in range(k + 1):
                # Case 1: not taking the current number
                forward[i][m] = forward[i-1][m].copy()
                # Case 2: taking the current number if possible
                if m >= 1:
                    prev_set = forward[i-1][m-1]
                    new_elements = {x | num for x in prev_set}
                    forward[i][m].update(new_elements)
        
        # Initialize backward DP table
        backward = [[set() for _ in range(k + 1)] for _ in range(n + 1)]
        backward[n][0].add(0)
        for i in range(n-1, -1, -1):
            num = nums[i]
            for m in range(k + 1):
                # Case 1: not taking the current number
                backward[i][m] = backward[i+1][m].copy()
                # Case 2: taking the current number if possible
                if m >= 1:
                    prev_set = backward[i+1][m-1]
                    new_elements = {x | num for x in prev_set}
                    backward[i][m].update(new_elements)
        
        max_val = 0
        # Iterate through possible split points
        for i in range(k, n - k + 1):
            a_set = forward[i][k]
            b_set = backward[i][k]
            for a in a_set:
                for b in b_set:
                    current = a ^ b
                    if current > max_val:
                        max_val = current
        return max_val
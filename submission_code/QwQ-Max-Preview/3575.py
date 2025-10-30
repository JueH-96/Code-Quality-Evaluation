from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n < 2 * k:
            return 0  # According to constraints, this should not happen
        
        # Initialize prefix OR DP
        prefix_or = [[set() for _ in range(k+1)] for _ in range(n+1)]
        prefix_or[0][0].add(0)
        for m in range(1, n+1):
            current_num = nums[m-1]
            # Copy previous state
            for j in range(k+1):
                prefix_or[m][j] = set(prefix_or[m-1][j])
            # Update with current_num for each j from 1 to k
            for j in range(1, k+1):
                for val in prefix_or[m-1][j-1]:
                    new_val = val | current_num
                    if new_val not in prefix_or[m][j]:
                        prefix_or[m][j].add(new_val)
        
        # Initialize suffix OR DP
        suffix_or = [[set() for _ in range(k+1)] for _ in range(n+1)]
        suffix_or[n][0].add(0)
        for m in range(n-1, -1, -1):
            current_num = nums[m]
            # Copy next state
            for j in range(k+1):
                suffix_or[m][j] = set(suffix_or[m+1][j])
            # Update with current_num for each j from 1 to k
            for j in range(1, k+1):
                for val in suffix_or[m+1][j-1]:
                    new_val = val | current_num
                    if new_val not in suffix_or[m][j]:
                        suffix_or[m][j].add(new_val)
        
        max_xor = 0
        # Iterate over all valid splits
        for m in range(k, n - k + 1):
            A = prefix_or[m][k]
            B = suffix_or[m][k]
            if not A or not B:
                continue
            current_max = max(a ^ b for a in A for b in B)
            if current_max > max_xor:
                max_xor = current_max
        
        return max_xor
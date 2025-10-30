from typing import List
import bisect

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # Precompute next greater index for each index.
        # For index i, r[i] = first index j > i s.t. nums[j] > nums[i], or n if none.
        next_greater = [n] * n
        stack = []
        for i in range(n):
            # while current element is greater than element at stack top, update next_greater
            while stack and nums[i] > nums[stack[-1]]:
                idx = stack.pop()
                next_greater[idx] = i
            stack.append(i)
        
        # Build positions mapping: value -> sorted list of indices where this value occurs.
        pos = {}
        for i, v in enumerate(nums):
            if v not in pos:
                pos[v] = []
            pos[v].append(i)
        
        result = 0
        # For each start index i, determine the valid right boundary (r = next greater index)
        # Then among indices in [i, r) that have the same value as nums[i],
        # each such index j (with j >= i) yields a valid subarray [i...j].
        # We count those j's.
        for i, v in enumerate(nums):
            r = next_greater[i]
            # Use binary search in pos[v] for the range [i, r)
            indices = pos[v]
            # left_bound = first occurrence >= i 
            l_idx = bisect.bisect_left(indices, i)
            # right_bound = first occurrence >= r
            r_idx = bisect.bisect_left(indices, r)
            count = r_idx - l_idx
            result += count
        
        return result
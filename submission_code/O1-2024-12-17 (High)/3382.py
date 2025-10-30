from typing import List
from bisect import bisect_left

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Step 1: Compute the index of the next strictly greater element for each position
        # If none exists, store n (meaning "no greater element to the right").
        next_greater = [n] * n
        stack = []
        # We'll process from right to left, using a standard "next greater element" approach.
        for i in reversed(range(n)):
            # Pop all elements that are <= nums[i], because they can't be the "strictly greater" for earlier indices
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            next_greater[i] = stack[-1] if stack else n
            stack.append(i)
        
        # Step 2: Group indices by their value.
        # For each distinct value v, we'll keep a sorted list of positions where nums[i] == v.
        positions_map = {}
        for i, val in enumerate(nums):
            positions_map.setdefault(val, []).append(i)
        
        # Step 3: Count all valid subarrays.
        # A subarray starting at index i (where nums[i] = v) can end at any index j in the same value-group
        # as long as j >= i and that position is still before the "next strictly greater" index for i.
        # That ensures no element in [i..j] is greater than nums[i].
        result = 0
        for val, pos_list in positions_map.items():
            # pos_list is already sorted by construction
            for i in range(len(pos_list)):
                start_idx = pos_list[i]
                # The subarray must end before next_greater[start_idx].
                limit = next_greater[start_idx]
                # Find how many positions in pos_list are strictly less than 'limit'.
                # Those positions can serve as valid end indices.
                # We use bisect_left to find the first index in pos_list where value >= limit,
                # then subtract 1 to get the largest valid position index.
                r = bisect_left(pos_list, limit) - 1
                # If r >= i, we have (r - i + 1) ways to pick an end index (including i == j for single-element).
                if r >= i:
                    result += (r - i + 1)
        
        return result
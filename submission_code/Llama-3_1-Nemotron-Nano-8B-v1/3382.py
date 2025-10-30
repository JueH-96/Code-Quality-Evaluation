from collections import defaultdict
import bisect
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        next_greater = [n] * n
        stack = []
        
        # Compute next_greater array using a monotonic stack
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                next_greater[i] = stack[-1]
            else:
                next_greater[i] = n  # No greater element found
            stack.append(i)
        
        # Map each number to the list of its positions
        pos_map = defaultdict(list)
        for idx, num in enumerate(nums):
            pos_map[num].append(idx)
        
        total = 0
        
        # For each number, process its positions to count valid subarrays
        for num in pos_map:
            positions = pos_map[num]
            for idx in range(len(positions)):
                i = positions[idx]
                ng = next_greater[i]  # Next greater element's index
                # Find the rightmost position j in positions where j < ng
                max_j_idx = bisect.bisect_left(positions, ng) - 1
                if max_j_idx >= idx:
                    count = max_j_idx - idx + 1
                    total += count
        
        return total
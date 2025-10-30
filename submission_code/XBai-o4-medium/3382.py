from bisect import bisect_right
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        next_greater = [None] * n
        stack = []
        
        for i in range(n):
            while stack and nums[i] > nums[stack[-1]]:
                popped = stack.pop()
                next_greater[popped] = i
            stack.append(i)
        
        groups = defaultdict(list)
        for idx, num in enumerate(nums):
            groups[num].append(idx)
        
        total = 0
        for val in groups:
            pos_list = groups[val]
            for j in range(len(pos_list)):
                ai = pos_list[j]
                # Compute R
                if next_greater[ai] is not None:
                    R = next_greater[ai] - 1
                else:
                    R = n - 1
                # Find k using bisect_right on pos_list
                k = bisect_right(pos_list, R)
                total += (k - j)
        
        return total
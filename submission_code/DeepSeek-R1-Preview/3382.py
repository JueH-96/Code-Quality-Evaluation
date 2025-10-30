from bisect import bisect_right
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Compute R for each index
        R = [n - 1] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                R[i] = stack[-1] - 1
            else:
                R[i] = n - 1
            stack.append(i)
        
        # Collect positions for each value
        pos = defaultdict(list)
        for idx, num in enumerate(nums):
            pos[num].append(idx)
        
        count = 0
        for v in pos:
            lst = pos[v]
            m = len(lst)
            for i in range(m):
                idx = lst[i]
                right_bound = R[idx]
                # Find the largest j in lst where j <= right_bound
                j = bisect_right(lst, right_bound) - 1
                count += (j - i + 1)
        
        return count
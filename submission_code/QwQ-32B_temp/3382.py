import bisect
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        next_greater_right = [n] * n
        stack = []
        
        # Compute next_greater_right using a monotonic stack
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                next_greater_right[i] = stack[-1]
            else:
                next_greater_right[i] = n
            stack.append(i)
        
        # Group indices by their value
        pos = defaultdict(list)
        for idx, num in enumerate(nums):
            pos[num].append(idx)
        
        total = 0
        
        for x in pos:
            lst = pos[x]
            m = len(lst)
            for i in range(m):
                a = lst[i]
                max_b = next_greater_right[a] - 1
                # Find the rightmost index j in lst[i..m-1] where lst[j] <= max_b
                j = bisect.bisect_right(lst, max_b, i, m) - 1
                if j >= i:
                    total += (j - i + 1)
        
        return total
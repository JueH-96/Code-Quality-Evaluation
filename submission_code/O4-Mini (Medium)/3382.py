from typing import List
from collections import defaultdict
from bisect import bisect_left

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        # L[i]: index of the previous strictly greater element to the left of i
        # R[i]: index of the next strictly greater element to the right of i
        L = [-1] * n
        R = [n] * n
        
        # Compute L using a monotonic decreasing stack
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            L[i] = stack[-1] if stack else -1
            stack.append(i)
        
        # Compute R using a monotonic decreasing stack (from the right)
        stack.clear()
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            R[i] = stack[-1] if stack else n
            stack.append(i)
        
        # Group positions by value
        pos_map = defaultdict(list)
        for i, v in enumerate(nums):
            pos_map[v].append(i)
        
        # For each start index i, count how many same-value positions j >= i
        # lie before the next greater element R[i].
        ans = 0
        for i, v in enumerate(nums):
            lst = pos_map[v]
            # first position in lst >= i
            lo = bisect_left(lst, i)
            # first position in lst >= R[i] (we want j < R[i])
            hi = bisect_left(lst, R[i])
            ans += hi - lo
        
        return ans
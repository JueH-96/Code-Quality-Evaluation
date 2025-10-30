import bisect
from collections import defaultdict

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        next_greater = [n] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] < nums[i]:
                idx = stack.pop()
                next_greater[idx] = i
            stack.append(i)
        
        d = defaultdict(list)
        for i, num in enumerate(nums):
            d[num].append(i)
        
        ans = 0
        for i in range(n):
            r = next_greater[i]
            arr = d[nums[i]]
            left_idx = bisect.bisect_left(arr, i)
            right_idx = bisect.bisect_right(arr, r - 1)
            ans += (right_idx - left_idx)
        
        return ans
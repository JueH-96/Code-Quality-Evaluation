import bisect
from collections import defaultdict

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        left_bound = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                left_bound[i] = stack[-1]
            else:
                left_bound[i] = -1
            stack.append(i)
        
        right_bound = [n] * n
        stack = []
        for i in range(n-1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                right_bound[i] = stack[-1]
            else:
                right_bound[i] = n
            stack.append(i)
        
        pos = defaultdict(list)
        for i in range(n):
            pos[nums[i]].append(i)
        
        total = 0
        for m in pos:
            p = pos[m]
            for i in p:
                j_max = right_bound[i] - 1
                l = bisect.bisect_left(p, i)
                r = bisect.bisect_right(p, j_max)
                cnt = r - l
                total += cnt
        
        return total
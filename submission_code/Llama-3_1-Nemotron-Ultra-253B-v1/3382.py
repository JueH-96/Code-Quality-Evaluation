from bisect import bisect_right
from collections import defaultdict
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        last_greater = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            if stack:
                last_greater[i] = stack[-1]
            else:
                last_greater[i] = -1
            stack.append(i)
        
        pos_dict = defaultdict(list)
        code_sum = 0
        for i in range(n):
            m = nums[i]
            lgp = last_greater[i]
            pos_list = pos_dict[m]
            count = len(pos_list) - bisect_right(pos_list, lgp)
            code_sum += count
            pos_list.append(i)
        
        return code_sum + n
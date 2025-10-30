from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        min_deque = deque()
        max_deque = deque()
        left = 0
        res = 0
        for right in range(n):
            while min_deque and nums[right] < nums[min_deque[-1]]:
                min_deque.pop()
            min_deque.append(right)
            while max_deque and nums[right] > nums[max_deque[-1]]:
                max_deque.pop()
            max_deque.append(right)
            while nums[max_deque[0]] - nums[min_deque[0]] > 2:
                if min_deque[0] == left:
                    min_deque.popleft()
                if max_deque[0] == left:
                    max_deque.popleft()
                left += 1
            res += right - left + 1
        return res
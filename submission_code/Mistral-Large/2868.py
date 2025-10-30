from typing import List
from collections import deque

class Solution:
    def continuousSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        left = 0
        count = 0
        q = deque()

        for right in range(n):
            while q and nums[right] < q[-1][0]:
                q.pop()
            q.append((nums[right], right))

            while q[0][0] - q[-1][0] > 2:
                if q[0][1] == left:
                    q.popleft()
                left += 1

            count += right - left + 1

        return count
from typing import List
from collections import deque

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        min_total = float('inf')
        for K in range(n):
            w = K + 1
            sum_min = self.sum_of_mins_circular(nums, w)
            total = sum_min + K * x
            if total < min_total:
                min_total = total
        return min_total
    
    def sum_of_mins_circular(self, arr, w):
        n = len(arr)
        duplicated = arr + arr
        q = deque()
        total = 0
        for i in range(len(duplicated)):
            # Remove elements from the end that are >= current
            while q and duplicated[i] <= duplicated[q[-1]]:
                q.pop()
            q.append(i)
            # Remove elements out of the window [i - w + 1, i]
            while q[0] < i - w + 1:
                q.popleft()
            # Check if the window is of size w and is one of the first n windows
            if i >= w - 1 and i < n + w - 1:
                total += duplicated[q[0]]
        return total
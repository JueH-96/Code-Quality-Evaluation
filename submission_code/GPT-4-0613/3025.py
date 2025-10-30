from typing import List
from collections import Counter
import heapq

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counter = Counter(nums)
        heap = []
        for num in counter:
            heapq.heappush(heap, (-num, counter[num]))
        operations = 0
        while heap:
            num, count = heapq.heappop(heap)
            num = -num
            if num <= target:
                operations += (target // num) * count
                target %= num
            else:
                operations += count
                if num // 2 >= target:
                    heapq.heappush(heap, (-(num // 2), count * 2))
            if target == 0:
                return operations
        return -1
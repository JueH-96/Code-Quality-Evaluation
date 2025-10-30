from typing import List
import heapq

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        nums.sort(reverse=True)
        heap = [-num for num in nums]
        heapq.heapify(heap)
        operations = 0
        while target > 0:
            max_num = -heapq.heappop(heap)
            if max_num > target:
                heapq.heappush(heap, -max_num // 2)
                operations += 1
            else:
                target -= max_num
        return operations if target == 0 else -1
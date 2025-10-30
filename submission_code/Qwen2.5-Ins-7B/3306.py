from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        marked = set()
        unmarked_sum = sum(nums)
        min_heap = [(nums[i], i) for i in range(len(nums))]
        import heapq
        heapq.heapify(min_heap)
        
        result = []
        for index, k in queries:
            if index not in marked:
                marked.add(index)
                unmarked_sum -= nums[index]
            while k > 0 and min_heap:
                value, i = heapq.heappop(min_heap)
                if i not in marked:
                    marked.add(i)
                    unmarked_sum -= value
                    k -= 1
            result.append(unmarked_sum)
        
        return result
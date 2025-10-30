import heapq
from typing import List

class Solution:
    def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        total_sum = sum(nums)
        marked = [False] * n
        marked_sum = 0
        
        # Create a min-heap of all elements, ordered by value and then index
        heap = []
        for i in range(n):
            heapq.heappush(heap, (nums[i], i))
        
        ans = []
        
        for index, k in queries:
            # Step a: mark the specified index if not already marked
            if not marked[index]:
                marked[index] = True
                marked_sum += nums[index]
            
            # Step b: mark k smallest unmarked elements
            count = 0
            while count < k and heap:
                val, idx = heapq.heappop(heap)
                if not marked[idx]:
                    marked[idx] = True
                    marked_sum += val
                    count += 1
        
            ans.append(total_sum - marked_sum)
        
        return ans
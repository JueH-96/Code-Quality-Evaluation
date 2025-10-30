import heapq
from typing import List

class Solution:
	def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
		total = sum(nums)
		n = len(nums)
		marked = [False] * n
		unmarked_sum = total
		heap = []
		for i, num in enumerate(nums):
			heapq.heappush(heap, (num, i))
		
		ans = []
		for query in queries:
			index = query[0]
			k = query[1]
			if not marked[index]:
				marked[index] = True
				unmarked_sum -= nums[index]
			
			count = 0
			while count < k and heap:
				while heap and marked[heap[0][1]]:
					heapq.heappop(heap)
				if not heap:
					break
				val, idx = heapq.heappop(heap)
				marked[idx] = True
				unmarked_sum -= val
				count += 1
			
			ans.append(unmarked_sum)
		
		return ans
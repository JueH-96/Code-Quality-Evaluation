import heapq
from typing import List

class Solution:
	def unmarkedSumArray(self, nums: List[int], queries: List[List[int]]) -> List[int]:
		n = len(nums)
		total = sum(nums)
		heap = [(nums[i], i) for i in range(n)]
		heapq.heapify(heap)
		marked = [False] * n
		current_sum = total
		ans = []
		
		for query in queries:
			idx = query[0]
			k = query[1]
			if not marked[idx]:
				marked[idx] = True
				current_sum -= nums[idx]
				
			count = 0
			while heap and count < k:
				val, i = heapq.heappop(heap)
				if marked[i]:
					continue
				marked[i] = True
				current_sum -= val
				count += 1
				
			ans.append(current_sum)
			
		return ans
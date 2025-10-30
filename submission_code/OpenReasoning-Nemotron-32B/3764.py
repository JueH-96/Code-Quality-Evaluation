import heapq
from typing import List

class Solution:
	def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
		n = len(grid)
		for i in range(n):
			grid[i].sort(reverse=True)
		
		heap = []
		for i in range(n):
			if limits[i] > 0 and grid[i]:
				heapq.heappush(heap, (-grid[i][0], i, 0))
		
		total_sum = 0
		count = 0
		while count < k and heap:
			val, i, idx = heapq.heappop(heap)
			total_sum += -val
			count += 1
			if idx + 1 < limits[i] and idx + 1 < len(grid[i]):
				heapq.heappush(heap, (-grid[i][idx+1], i, idx+1))
		
		return total_sum
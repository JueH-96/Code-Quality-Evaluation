import heapq
from typing import List

class Solution:
	def maxSum(self, grid: List[List[int]], limits: List[int], k: int) -> int:
		n = len(grid)
		rows = [sorted(row, reverse=True) for row in grid]
		current_index = [0] * n
		heap = []
		for i in range(n):
			if limits[i] > 0 and current_index[i] < len(rows[i]):
				heapq.heappush(heap, (-rows[i][current_index[i]], i))
		
		total = 0
		count = 0
		while count < k and heap:
			neg_val, i = heapq.heappop(heap)
			total += -neg_val
			count += 1
			current_index[i] += 1
			if current_index[i] < limits[i] and current_index[i] < len(rows[i]):
				heapq.heappush(heap, (-rows[i][current_index[i]], i))
		
		return total
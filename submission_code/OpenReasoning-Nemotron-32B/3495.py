import heapq
from typing import List

class Solution:
	def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
		heap = []
		results = []
		for x, y in queries:
			d = abs(x) + abs(y)
			if len(heap) < k:
				heapq.heappush(heap, -d)
				if len(heap) == k:
					results.append(-heap[0])
				else:
					results.append(-1)
			else:
				if d < -heap[0]:
					heapq.heappushpop(heap, -d)
					results.append(-heap[0])
				else:
					results.append(-heap[0])
		return results
import heapq

class Solution:
	def resultsArray(self, queries: List[List[int]], k: int) -> List[int]:
		heap = []
		res = []
		for x, y in queries:
			dist = abs(x) + abs(y)
			if len(heap) < k:
				heapq.heappush(heap, -dist)
				if len(heap) < k:
					res.append(-1)
				else:
					res.append(-heap[0])
			else:
				heapq.heappushpop(heap, -dist)
				res.append(-heap[0])
		return res
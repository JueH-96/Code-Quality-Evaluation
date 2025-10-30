import heapq

class Solution:
	def maxDistance(self, s: str, k: int) -> int:
		x, y = 0, 0
		ans = 0
		heap = []
		total_improve = 0
		
		for c in s:
			x0, y0 = x, y
			if c == 'N':
				y += 1
			elif c == 'S':
				y -= 1
			elif c == 'E':
				x += 1
			else:
				x -= 1
			
			imp = (abs(x0) + abs(y0) + 1) - (abs(x) + abs(y))
			heapq.heappush(heap, imp)
			total_improve += imp
			
			if len(heap) > k:
				smallest = heapq.heappop(heap)
				total_improve -= smallest
				
			candidate = abs(x) + abs(y) + total_improve
			if candidate > ans:
				ans = candidate
				
		return ans
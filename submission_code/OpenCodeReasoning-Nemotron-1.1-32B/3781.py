from typing import List

class Solution:
	def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
		points_sorted = sorted(points, key=lambda x: (x[0], x[1]))
		
		def check(d):
			selected = []
			for p in points_sorted:
				for s in selected:
					if abs(s[0] - p[0]) + abs(s[1] - p[1]) < d:
						break
				else:
					selected.append(p)
					if len(selected) == k:
						return True
			return False
		
		low, high = 0, 2 * side
		while low <= high:
			mid = (low + high) // 2
			if check(mid):
				low = mid + 1
			else:
				high = mid - 1
		return high
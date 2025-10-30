class Solution:
	def maxScore(self, points: List[int], m: int) -> int:
		n = len(points)
		lo, hi = 0, m * max(points)
		while lo < hi:
			mid = (lo + hi + 1) // 2
			total_visits = 0
			for i in range(n):
				if points[i] > 0:
					total_visits += (mid + points[i] - 1) // points[i]
			if total_visits > m:
				hi = mid - 1
			else:
				lo = mid
		return lo
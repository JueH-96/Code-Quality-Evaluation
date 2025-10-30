class Solution:
	def maxScore(self, points: List[int], m: int) -> int:
		n = len(points)
		if m < n:
			return 0
		
		max_point = max(points)
		low, high = 0, m * max_point + 1
		
		while low < high:
			mid = (low + high + 1) // 2
			total_visits = 0
			for p in points:
				if mid == 0:
					ceil_val = 0
				else:
					ceil_val = (mid + p - 1) // p
				total_visits += ceil_val
				if total_visits > m:
					break
					
			if total_visits <= m:
				low = mid
			else:
				high = mid - 1
				
		return low
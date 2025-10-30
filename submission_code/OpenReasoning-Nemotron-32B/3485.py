class Solution:
	def maxPossibleScore(self, start: List[int], d: int) -> int:
		start.sort()
		n = len(start)
		low, high = 0, start[-1] + d - start[0]
		
		def feasible(x):
			last = start[0]
			for i in range(1, n):
				if last + x > start[i] + d:
					return False
				last = max(last + x, start[i])
			return True
		
		while low <= high:
			mid = (low + high) // 2
			if feasible(mid):
				low = mid + 1
			else:
				high = mid - 1
		return high
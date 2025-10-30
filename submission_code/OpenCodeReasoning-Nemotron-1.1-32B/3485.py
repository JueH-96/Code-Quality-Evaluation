class Solution:
	def maxPossibleScore(self, start: List[int], d: int) -> int:
		if not start:
			return 0
		sorted_start = sorted(start)
		n = len(start)
		low = 0
		high = sorted_start[-1] + d - sorted_start[0]
		
		def check(x):
			last = -10**18
			for a in sorted_start:
				candidate = max(a, last + x)
				if candidate > a + d:
					return False
				last = candidate
			return True
		
		while low <= high:
			mid = (low + high) // 2
			if check(mid):
				low = mid + 1
			else:
				high = mid - 1
				
		return high
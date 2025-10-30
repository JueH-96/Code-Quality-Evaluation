class Solution:
	def maximumLength(self, nums: List[int]) -> int:
		e = 0
		o = 0
		dp0 = 0
		dp1 = 0
		max_alt = 0
		for num in nums:
			if num % 2 == 0:
				e += 1
				dp0 = dp1 + 1
				if dp0 > max_alt:
					max_alt = dp0
			else:
				o += 1
				dp1 = dp0 + 1
				if dp1 > max_alt:
					max_alt = dp1
		return max(e, o, max_alt)
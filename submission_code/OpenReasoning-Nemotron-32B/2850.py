from functools import lru_cache

class Solution:
	def longestString(self, x: int, y: int, z: int) -> int:
		@lru_cache(maxsize=None)
		def dp(a, b, c, last):
			res = 0
			if a > 0 and last != 'A':
				res = max(res, 2 + dp(a-1, b, c, 'A'))
			if b > 0 and last != 'B':
				res = max(res, 2 + dp(a, b-1, c, 'B'))
			if c > 0:
				res = max(res, 2 + dp(a, b, c-1, 'B'))
			return res
		
		return dp(x, y, z, '')
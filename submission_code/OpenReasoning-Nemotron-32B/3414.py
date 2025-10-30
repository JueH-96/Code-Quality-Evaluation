from functools import lru_cache

class Solution:
	def waysToReachStair(self, k: int) -> int:
		if k == 0:
			return 2
		if k > 1000:
			return 0
		
		bound_pos = 2 * k + 10000
		
		@lru_cache(maxsize=None)
		def dfs(pos, jump, last_down):
			if pos == k:
				return 1
			if pos < 0 or pos > bound_pos or jump > 60:
				return 0
			ways = 0
			if not last_down and pos >= 1:
				ways += dfs(pos - 1, jump, True)
			new_pos = pos + (1 << jump)
			ways += dfs(new_pos, jump + 1, False)
			return ways
		
		return dfs(1, 0, False)
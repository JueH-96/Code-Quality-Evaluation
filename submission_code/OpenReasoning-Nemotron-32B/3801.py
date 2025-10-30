import functools

class Solution:
	def beautifulNumbers(self, l: int, r: int) -> int:
		return self.count(r) - self.count(l-1)
	
	def count(self, n: int) -> int:
		if n == 0:
			return 0
		digits = list(map(int, str(n)))
		L = len(digits)
		max_sum = 9 * L
		total_count = 0
		for s in range(1, min(91, max_sum + 1)):
			@functools.lru_cache(maxsize=None)
			def dp(pos, tight, started, curr_sum, curr_mod):
				if pos == L:
					if started and curr_sum == s and curr_mod == 0:
						return 1
					return 0
				total = 0
				low = 0
				high = digits[pos] if tight else 9
				for d in range(low, high + 1):
					new_tight = tight and (d == high)
					new_started = started or (d > 0)
					if not started and d == 0:
						new_sum = 0
						new_mod = 1
					else:
						new_sum = curr_sum + d
						if new_sum > s:
							continue
						if d == 0:
							new_mod = 0
						else:
							new_mod = (curr_mod * d) % s
					total += dp(pos + 1, new_tight, new_started, new_sum, new_mod)
				return total
			res = dp(0, True, False, 0, 1)
			total_count += res
		return total_count
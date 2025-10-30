class Solution:
	def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
		def count_up_to(n, k):
			if n < 0:
				return 0
			s = str(n)
			n_digits = len(s)
			from functools import lru_cache
			@lru_cache(maxsize=None)
			def dp(pos, tight, started, rem, even, odd):
				if pos == n_digits:
					if started and even == odd and rem == 0:
						return 1
					return 0
				total = 0
				limit = int(s[pos]) if tight else 9
				for d in range(0, limit + 1):
					next_tight = tight and (d == limit)
					next_started = started or (d != 0)
					if not started and d == 0:
						total += dp(pos + 1, next_tight, next_started, (rem * 10 + d) % k, even, odd)
					else:
						if d % 2 == 0:
							next_even = even + 1
							next_odd = odd
						else:
							next_even = even
							next_odd = odd + 1
						total += dp(pos + 1, next_tight, next_started, (rem * 10 + d) % k, next_even, next_odd)
				return total
			return dp(0, True, False, 0, 0, 0)
		
		return count_up_to(high, k) - count_up_to(low - 1, k)
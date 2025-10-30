class Solution:
	def countSteppingNumbers(self, low: str, high: str) -> int:
		mod = 10**9 + 7
		
		def subtract_one(s):
			arr = [int(char) for char in s]
			n = len(arr)
			i = n - 1
			while i >= 0 and arr[i] == 0:
				arr[i] = 9
				i -= 1
			if i < 0:
				return "0"
			arr[i] -= 1
			result = ''.join(str(x) for x in arr)
			j = 0
			while j < len(result) and result[j] == '0':
				j += 1
			if j == len(result):
				return "0"
			return result[j:]
		
		def count_up_to(s):
			n = len(s)
			from functools import lru_cache
			@lru_cache(maxsize=None)
			def dp(pos, tight, started, last):
				if pos == n:
					return 1 if started else 0
				total = 0
				high_bound = int(s[pos]) if tight else 9
				for d in range(0, high_bound + 1):
					new_tight = tight and (d == high_bound)
					if not started:
						if d == 0:
							total = (total + dp(pos + 1, new_tight, False, -1)) % mod
						else:
							total = (total + dp(pos + 1, new_tight, True, d)) % mod
					else:
						if abs(d - last) == 1:
							total = (total + dp(pos + 1, new_tight, True, d)) % mod
				return total % mod
			return dp(0, True, False, -1)
		
		low_minus = subtract_one(low)
		count_high = count_up_to(high)
		count_low_minus = count_up_to(low_minus)
		result = (count_high - count_low_minus) % mod
		return result
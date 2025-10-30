MOD = 10**9 + 7

class Solution:
	def countSteppingNumbers(self, low: str, high: str) -> int:
		def subtract_one(s):
			arr = [int(c) for c in s]
			n = len(arr)
			i = n - 1
			while i >= 0:
				if arr[i] > 0:
					arr[i] -= 1
					break
				else:
					arr[i] = 9
					i -= 1
			res = ''.join(str(x) for x in arr).lstrip('0')
			return res if res != '' else '0'
		
		def count_up_to(s):
			n = len(s)
			from functools import lru_cache
			@lru_cache(maxsize=None)
			def dfs(pos, tight, last):
				if pos == n:
					return 1 if last != -1 else 0
				total = 0
				low_bound = 0
				high_bound = int(s[pos]) if tight else 9
				for d in range(low_bound, high_bound + 1):
					new_tight = tight and (d == high_bound)
					if last == -1:
						if d == 0:
							total = (total + dfs(pos + 1, new_tight, -1)) % MOD
						else:
							total = (total + dfs(pos + 1, new_tight, d)) % MOD
					else:
						if abs(d - last) == 1:
							total = (total + dfs(pos + 1, new_tight, d)) % MOD
				return total % MOD
			return dfs(0, True, -1)
		
		low_minus = subtract_one(low)
		total_high = count_up_to(high)
		total_low_minus = count_up_to(low_minus)
		result = (total_high - total_low_minus) % MOD
		return result
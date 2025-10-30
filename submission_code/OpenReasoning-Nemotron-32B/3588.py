MOD = 10**9 + 7

class Solution:
	def countWinningSequences(self, s: str) -> int:
		n = len(s)
		if n == 0:
			return 0
		
		def get_delta(a, b):
			if a == 'F':
				if b == 0:
					return 0
				elif b == 1:
					return 1
				else:
					return -1
			elif a == 'W':
				if b == 0:
					return -1
				elif b == 1:
					return 0
				else:
					return 1
			else:
				if b == 0:
					return 1
				elif b == 1:
					return -1
				else:
					return 0
		
		dp = [[0] * (2 * n + 1) for _ in range(3)]
		
		first_char = s[0]
		for b in range(3):
			delta = get_delta(first_char, b)
			d_index = delta + n
			if 0 <= d_index <= 2 * n:
				dp[b][d_index] = (dp[b][d_index] + 1) % MOD
		
		for i in range(1, n):
			a_char = s[i]
			new_dp = [[0] * (2 * n + 1) for _ in range(3)]
			for last in range(3):
				for d_index in range(2 * n + 1):
					count = dp[last][d_index]
					if count == 0:
						continue
					for nxt in range(3):
						if nxt == last:
							continue
						delta_val = get_delta(a_char, nxt)
						new_d_index = d_index + delta_val
						if 0 <= new_d_index <= 2 * n:
							new_dp[nxt][new_d_index] = (new_dp[nxt][new_d_index] + count) % MOD
			dp = new_dp
		
		total = 0
		for last in range(3):
			for d_index in range(n + 1, 2 * n + 1):
				total = (total + dp[last][d_index]) % MOD
		return total
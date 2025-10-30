MOD = 10**9 + 7

class Solution:
	def countWinningSequences(self, s: str) -> int:
		n = len(s)
		if n == 0:
			return 0
		
		change_arr = [
			[0, 1, -1],
			[-1, 0, 1],
			[1, -1, 0]
		]
		
		dp = [[0] * (2 * n + 1) for _ in range(3)]
		
		a0 = s[0]
		a0_index = 0 if a0 == 'F' else (1 if a0 == 'W' else 2)
		for b in range(3):
			ch = change_arr[a0_index][b]
			d_index0 = ch + n
			if 0 <= d_index0 < 2 * n + 1:
				dp[b][d_index0] = (dp[b][d_index0] + 1) % MOD
		
		for i in range(1, n):
			new_dp = [[0] * (2 * n + 1) for _ in range(3)]
			a_char = s[i]
			a_index = 0 if a_char == 'F' else (1 if a_char == 'W' else 2)
			for prev in range(3):
				for d_index in range(2 * n + 1):
					cnt = dp[prev][d_index]
					if cnt == 0:
						continue
					for b in range(3):
						if b == prev:
							continue
						ch = change_arr[a_index][b]
						current_d = d_index - n
						new_d = current_d + ch
						new_d_index = new_d + n
						if 0 <= new_d_index < 2 * n + 1:
							new_dp[b][new_d_index] = (new_dp[b][new_d_index] + cnt) % MOD
			dp = new_dp
		
		total = 0
		for move in range(3):
			for d_index in range(n + 1, 2 * n + 1):
				total = (total + dp[move][d_index]) % MOD
		return total
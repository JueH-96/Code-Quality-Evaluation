class Solution:
	def largestPalindrome(self, n: int, k: int) -> str:
		if n == 0:
			return ""
		m = (n + 1) // 2
		
		power10 = [0] * n
		power10[0] = 1 % k
		for i in range(1, n):
			power10[i] = (power10[i-1] * 10) % k
		
		factors = [0] * m
		for i in range(m):
			if n % 2 == 0:
				factors[i] = (power10[i] + power10[n-1-i]) % k
			else:
				if i < m-1:
					factors[i] = (power10[i] + power10[n-1-i]) % k
				else:
					factors[i] = power10[i]
		
		dp = [False] * k
		dp[0] = True
		
		parent = [[None] * k for _ in range(m)]
		
		for i in range(m):
			next_state = [None] * k
			for r in range(k):
				if not dp[r]:
					continue
				for d in range(9, -1, -1):
					if i == 0 and d == 0:
						continue
					new_r = (r + d * factors[i]) % k
					if next_state[new_r] is None or d > next_state[new_r][0]:
						next_state[new_r] = (d, r)
			new_dp = [False] * k
			for residue in range(k):
				if next_state[residue] is not None:
					new_dp[residue] = True
					parent[i][residue] = next_state[residue]
			dp = new_dp
		
		if not dp[0]:
			return ""
		
		digits = [0] * m
		current_residue = 0
		for i in range(m-1, -1, -1):
			d, prev_r = parent[i][current_residue]
			digits[i] = d
			current_residue = prev_r
		
		s = ''.join(str(d) for d in digits)
		if n % 2 == 0:
			return s + s[::-1]
		else:
			return s + s[:-1][::-1]
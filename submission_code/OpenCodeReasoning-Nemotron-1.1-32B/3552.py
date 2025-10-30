class Solution:
	def largestPalindrome(self, n: int, k: int) -> str:
		h = (n + 1) // 2
		
		power = [1] * (n + 1)
		for i in range(1, n + 1):
			power[i] = (power[i - 1] * 10) % k
		
		coeff = [0] * h
		if n % 2 == 0:
			for i in range(h):
				coeff[i] = (power[n - 1 - i] + power[i]) % k
		else:
			for i in range(h - 1):
				coeff[i] = (power[n - 1 - i] + power[i]) % k
			coeff[h - 1] = power[h - 1]
		
		dp = [[False] * k for _ in range(h + 1)]
		dp[h][0] = True
		
		for i in range(h - 1, -1, -1):
			for r in range(k):
				found = False
				for d in range(0, 10):
					new_r = (r - d * coeff[i]) % k
					if dp[i + 1][new_r]:
						found = True
						break
				dp[i][r] = found
		
		res = []
		current_r = 0
		for i in range(h):
			found = False
			for d in range(9, -1, -1):
				if i == 0 and d == 0:
					continue
				new_r = (current_r - d * coeff[i]) % k
				if dp[i + 1][new_r]:
					res.append(str(d))
					current_r = new_r
					found = True
					break
			if not found:
				return ""
		
		first_half = ''.join(res)
		if n % 2 == 0:
			ans = first_half + first_half[::-1]
		else:
			ans = first_half + first_half[:-1][::-1]
		return ans
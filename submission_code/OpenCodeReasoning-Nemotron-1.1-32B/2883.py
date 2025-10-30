class Solution:
	def minimumBeautifulSubstrings(self, s: str) -> int:
		max_val = (1 << 15) - 1
		power5_set = set()
		num = 1
		while num <= max_val:
			bin_rep = bin(num)[2:]
			power5_set.add(bin_rep)
			num *= 5
		
		n = len(s)
		dp = [10**9] * (n + 1)
		dp[0] = 0
		
		for i in range(1, n + 1):
			for j in range(i - 1, -1, -1):
				if i - j > 14:
					break
				if s[j] == '0':
					continue
				t = s[j:i]
				if t in power5_set:
					if dp[j] != 10**9:
						if dp[j] + 1 < dp[i]:
							dp[i] = dp[j] + 1
		
		return dp[n] if dp[n] != 10**9 else -1
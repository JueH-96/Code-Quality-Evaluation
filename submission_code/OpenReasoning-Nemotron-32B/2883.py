class Solution:
	def minimumBeautifulSubstrings(self, s: str) -> int:
		beautiful_set = set()
		power5 = 1
		while power5 < (1 << 16):
			bin_rep = bin(power5)[2:]
			if len(bin_rep) > 15:
				break
			beautiful_set.add(bin_rep)
			power5 *= 5
		
		n = len(s)
		dp = [float('inf')] * (n + 1)
		dp[0] = 0
		
		for i in range(1, n + 1):
			for j in range(i - 1, -1, -1):
				if s[j] == '0' and j < i - 1:
					continue
				substr = s[j:i]
				if substr in beautiful_set:
					if dp[j] != float('inf'):
						if dp[j] + 1 < dp[i]:
							dp[i] = dp[j] + 1
		
		return dp[n] if dp[n] != float('inf') else -1
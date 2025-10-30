class Solution:
	def minimumSubstringsInPartition(self, s: str) -> int:
		n = len(s)
		dp = [10**9] * (n + 1)
		dp[0] = 0
		
		for i in range(1, n + 1):
			freq = [0] * 26
			max_freq = 0
			distinct = 0
			for j in range(i - 1, -1, -1):
				char = s[j]
				idx = ord(char) - ord('a')
				old = freq[idx]
				new_val = old + 1
				freq[idx] = new_val
				
				if old == 0:
					distinct += 1
				if new_val > max_freq:
					max_freq = new_val
				
				length = i - j
				if max_freq * distinct == length:
					if j == 0:
						dp[i] = min(dp[i], 1)
					else:
						dp[i] = min(dp[i], dp[j] + 1)
		
		return dp[n]
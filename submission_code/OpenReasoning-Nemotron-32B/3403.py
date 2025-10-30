class Solution:
	def minimumSubstringsInPartition(self, s: str) -> int:
		n = len(s)
		dp = [float('inf')] * (n + 1)
		dp[0] = 0
		
		for i in range(n):
			if dp[i] == float('inf'):
				continue
			freq = [0] * 26
			distinct_count = 0
			max_freq = 0
			for j in range(i, n):
				idx = ord(s[j]) - ord('a')
				freq[idx] += 1
				if freq[idx] == 1:
					distinct_count += 1
				if freq[idx] > max_freq:
					max_freq = freq[idx]
				total_length = j - i + 1
				if max_freq * distinct_count == total_length:
					if dp[j + 1] > dp[i] + 1:
						dp[j + 1] = dp[i] + 1
		return dp[n]
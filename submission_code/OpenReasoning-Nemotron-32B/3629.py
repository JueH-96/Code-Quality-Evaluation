class Solution:
	def lengthAfterTransformations(self, s: str, t: int) -> int:
		mod_val = 10**9 + 7
		dp = [1] * 26
		for _ in range(t):
			new_dp = [0] * 26
			for i in range(25):
				new_dp[i] = dp[i+1]
			new_dp[25] = (dp[0] + dp[1]) % mod_val
			dp = new_dp
		total = 0
		for char in s:
			idx = ord(char) - ord('a')
			total = (total + dp[idx]) % mod_val
		return total
class Solution:
	def removeAlmostEqualCharacters(self, word: str) -> int:
		n = len(word)
		dp = [[10**9] * 26 for _ in range(n)]
		
		for c in range(26):
			char = chr(ord('a') + c)
			cost = 0 if char == word[0] else 1
			dp[0][c] = cost
		
		for i in range(1, n):
			for c in range(26):
				for p in range(26):
					if p == c or abs(p - c) == 1:
						continue
					current_char = chr(ord('a') + c)
					cost_here = 0 if current_char == word[i] else 1
					total_cost = dp[i-1][p] + cost_here
					if total_cost < dp[i][c]:
						dp[i][c] = total_cost
		
		return min(dp[n-1])
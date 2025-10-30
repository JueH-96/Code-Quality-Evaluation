class Solution:
	def minimizeConcatenatedLength(self, words: List[str]) -> int:
		n = len(words)
		if n == 0:
			return 0
		INF = 10**9
		dp = [[INF] * 26 for _ in range(26)]
		
		first_word = words[0]
		start0 = ord(first_word[0]) - ord('a')
		end0 = ord(first_word[-1]) - ord('a')
		dp[start0][end0] = len(first_word)
		
		for i in range(1, n):
			w = words[i]
			a = ord(w[0]) - ord('a')
			b = ord(w[-1]) - ord('a')
			L = len(w)
			new_dp = [[INF] * 26 for _ in range(26)]
			
			for x in range(26):
				for y in range(26):
					if dp[x][y] == INF:
						continue
					if y == a:
						new_len1 = dp[x][y] + L - 1
						if new_len1 < new_dp[x][b]:
							new_dp[x][b] = new_len1
					else:
						new_len1 = dp[x][y] + L
						if new_len1 < new_dp[x][b]:
							new_dp[x][b] = new_len1
					
					if b == x:
						new_len2 = dp[x][y] + L - 1
						if new_len2 < new_dp[a][y]:
							new_dp[a][y] = new_len2
					else:
						new_len2 = dp[x][y] + L
						if new_len2 < new_dp[a][y]:
							new_dp[a][y] = new_len2
			dp = new_dp
		
		ans = min(min(row) for row in dp)
		return ans
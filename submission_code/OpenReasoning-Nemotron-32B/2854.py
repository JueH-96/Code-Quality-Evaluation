class Solution:
	def minimizeConcatenatedLength(self, words: List[str]) -> int:
		n = len(words)
		INF = 10**9
		dp = [[INF] * 26 for _ in range(26)]
		
		first_word = words[0]
		start_char = ord(first_word[0]) - ord('a')
		end_char = ord(first_word[-1]) - ord('a')
		dp[start_char][end_char] = len(first_word)
		
		for i in range(1, n):
			word = words[i]
			s_char = ord(word[0]) - ord('a')
			t_char = ord(word[-1]) - ord('a')
			L = len(word)
			new_dp = [[INF] * 26 for _ in range(26)]
			
			for a in range(26):
				for b in range(26):
					if dp[a][b] == INF:
						continue
					# Option 1: current string followed by the new word
					if b == s_char:
						new_length1 = dp[a][b] + L - 1
					else:
						new_length1 = dp[a][b] + L
					if new_length1 < new_dp[a][t_char]:
						new_dp[a][t_char] = new_length1
					
					# Option 2: new word followed by the current string
					if t_char == a:
						new_length2 = dp[a][b] + L - 1
					else:
						new_length2 = dp[a][b] + L
					if new_length2 < new_dp[s_char][b]:
						new_dp[s_char][b] = new_length2
			dp = new_dp
		
		ans = INF
		for a in range(26):
			for b in range(26):
				if dp[a][b] < ans:
					ans = dp[a][b]
		return ans
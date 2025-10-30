class Solution:
	def longestPalindrome(self, s: str, t: str) -> int:
		n = len(s)
		m = len(t)
		t_rev = t[::-1]
		
		def get_longest_palindrome(x):
			if not x:
				return 0
			max_len = 1
			length = len(x)
			for i in range(length):
				l, r = i, i
				while l >= 0 and r < length and x[l] == x[r]:
					max_len = max(max_len, r - l + 1)
					l -= 1
					r += 1
			for i in range(length - 1):
				l, r = i, i + 1
				while l >= 0 and r < length and x[l] == x[r]:
					max_len = max(max_len, r - l + 1)
					l -= 1
					r += 1
			return max_len
		
		candidate_single = max(get_longest_palindrome(s), get_longest_palindrome(t))
		
		dp = [[0] * (m + 1) for _ in range(n + 1)]
		max_even = 0
		for i in range(1, n + 1):
			for j in range(1, m + 1):
				if s[i - 1] == t_rev[j - 1]:
					dp[i][j] = dp[i - 1][j - 1] + 1
					if dp[i][j] > max_even:
						max_even = dp[i][j]
				else:
					dp[i][j] = 0
		candidate_even = 2 * max_even
		
		candidate_center = 0
		for i in range(n):
			for j in range(m + 1):
				L = dp[i][j]
				candidate_center = max(candidate_center, 2 * L + 1)
				
		for j in range(m):
			k = m - 1 - j
			for i in range(n + 1):
				L = dp[i][k]
				candidate_center = max(candidate_center, 2 * L + 1)
				
		return max(candidate_single, candidate_even, candidate_center)
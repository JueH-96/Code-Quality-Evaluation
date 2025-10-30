class Solution:
	def minimizedStringLength(self, s: str) -> int:
		n = len(s)
		dp = [[0] * n for _ in range(n)]
		
		for i in range(n):
			dp[i][i] = 1
		
		for length in range(2, n + 1):
			for i in range(n - length + 1):
				j = i + length - 1
				best = length
				for k in range(i, j + 1):
					left_index = None
					p = k - 1
					while p >= i:
						if s[p] == s[k]:
							left_index = p
							break
						p -= 1
					
					right_index = None
					p = k + 1
					while p <= j:
						if s[p] == s[k]:
							right_index = p
							break
						p += 1
					
					if left_index is not None and right_index is not None:
						part1 = dp[i][left_index - 1] if left_index - 1 >= i else 0
						part2 = dp[left_index + 1][right_index - 1] if left_index + 1 <= right_index - 1 else 0
						part3 = dp[right_index + 1][j] if right_index + 1 <= j else 0
						total = part1 + part2 + part3
						if total < best:
							best = total
					elif left_index is not None:
						part1 = dp[i][left_index - 1] if left_index - 1 >= i else 0
						part2 = dp[left_index + 1][j] if left_index + 1 <= j else 0
						total = part1 + part2
						if total < best:
							best = total
					elif right_index is not None:
						part1 = dp[i][right_index - 1] if right_index - 1 >= i else 0
						part2 = dp[right_index + 1][j] if right_index + 1 <= j else 0
						total = part1 + part2
						if total < best:
							best = total
				dp[i][j] = best
		return dp[0][n - 1]
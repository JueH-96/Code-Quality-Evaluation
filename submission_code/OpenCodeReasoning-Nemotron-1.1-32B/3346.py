class Solution:
	def getSmallestString(self, s: str, k: int) -> str:
		n = len(s)
		dp = [[''] * (k+1) for _ in range(n+1)]
		
		for i in range(n-1, -1, -1):
			for k_rem in range(0, k+1):
				best_char = None
				best_rest = None
				for c in "abcdefghijklmnopqrstuvwxyz":
					a_val = ord(s[i])
					b_val = ord(c)
					diff = abs(a_val - b_val)
					cost = min(diff, 26 - diff)
					if cost > k_rem:
						continue
					rest_str = dp[i+1][k_rem - cost]
					if best_char is None:
						best_char = c
						best_rest = rest_str
					else:
						if c < best_char:
							best_char = c
							best_rest = rest_str
						elif c > best_char:
							break
				dp[i][k_rem] = best_char + best_rest
		
		return dp[0][k]
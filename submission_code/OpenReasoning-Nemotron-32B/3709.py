class Solution:
	def hasSpecialSubstring(self, s: str, k: int) -> bool:
		n = len(s)
		for i in range(n - k + 1):
			if s[i:i+k] == s[i] * k:
				left_ok = (i == 0) or (s[i-1] != s[i])
				right_ok = (i + k == n) or (s[i+k] != s[i])
				if left_ok and right_ok:
					return True
		return False
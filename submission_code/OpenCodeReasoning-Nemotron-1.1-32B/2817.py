class Solution:
	def minimumCost(self, s: str) -> int:
		n = len(s)
		if n == 0:
			return 0
		ans = 0
		for i in range(n - 1):
			if s[i] != s[i + 1]:
				ans += i + 1
		return ans
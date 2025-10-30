class Solution:
	def maxOperations(self, s: str) -> int:
		ans = 0
		zero = 0
		for c in s:
			if c == '0':
				zero += 1
			else:
				if zero > 0:
					ans += 1
					zero -= 1
		return ans
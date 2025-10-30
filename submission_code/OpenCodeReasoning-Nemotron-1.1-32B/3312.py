class Solution:
	def countKeyChanges(self, s: str) -> int:
		count = 0
		n = len(s)
		for i in range(n - 1):
			if s[i].lower() != s[i+1].lower():
				count += 1
		return count
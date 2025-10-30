class Solution:
	def minChanges(self, n: int, k: int) -> int:
		if n < k:
			return -1
		
		changes = 0
		while n or k:
			if (k & 1) > (n & 1):
				return -1
			if (n & 1) > (k & 1):
				changes += 1
			n >>= 1
			k >>= 1
		return changes
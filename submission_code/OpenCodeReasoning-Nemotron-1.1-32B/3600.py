class Solution:
	def kthCharacter(self, k: int) -> str:
		total = 1
		while total < k:
			total *= 2
		shift = 0
		while total > 1:
			half = total // 2
			if k > half:
				k -= half
				shift += 1
			total = half
		return chr(ord('a') + shift)
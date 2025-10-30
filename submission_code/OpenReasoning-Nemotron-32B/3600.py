class Solution:
	def kthCharacter(self, k: int) -> str:
		L = 1
		while L < k:
			L *= 2
		
		transformations = 0
		seg_len = L
		pos = k
		while seg_len > 1:
			half = seg_len // 2
			if pos > half:
				pos -= half
				transformations += 1
			seg_len = half
		
		return chr(ord('a') + transformations % 26)
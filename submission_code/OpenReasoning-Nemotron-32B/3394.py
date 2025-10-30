class Solution:
	def minEnd(self, n: int, x: int) -> int:
		if n == 1:
			return x
		t = (n - 1).bit_length()
		positions = []
		j = 0
		while len(positions) < t:
			if (x >> j) & 1 == 0:
				positions.append(j)
			j += 1
		s = 0
		for i in range(t):
			if (n - 1) & (1 << i):
				s |= (1 << positions[i])
		return x + s
class Solution:
	def minEnd(self, n: int, x: int) -> int:
		if x == 0:
			return n - 1
		m = x.bit_length() - 1
		k = 0
		while (x >> k) & 1:
			k += 1
		
		if k > m:
			group_size = 1
		else:
			group_size = 1 << (m - k)
		
		if n <= group_size:
			return x + (1 << k) * (n - 1)
		else:
			i = n - 1
			remaining = i - group_size
			group_index = remaining // group_size
			offset = remaining % group_size
			return x + (1 << (m + 1 + group_index)) + offset
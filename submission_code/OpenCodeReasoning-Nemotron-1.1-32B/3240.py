class Solution:
	def findMaximumNumber(self, k: int, x: int) -> int:
		def count_set_bits(n, p):
			if n < 0:
				return 0
			period = 1 << (p + 1)
			full_cycles = (n + 1) // period
			rem = (n + 1) % period
			return full_cycles * (1 << p) + max(0, rem - (1 << p))
		
		def F(n):
			if n <= 0:
				return 0
			total = 0
			j = x
			while j - 1 < n.bit_length():
				p = j - 1
				total += count_set_bits(n, p)
				j += x
			return total
		
		low, high = 0, 10**18
		while low <= high:
			mid = (low + high) // 2
			if F(mid) <= k:
				low = mid + 1
			else:
				high = mid - 1
		return high
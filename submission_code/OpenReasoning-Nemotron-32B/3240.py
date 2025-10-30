class Solution:
	def findMaximumNumber(self, k: int, x: int) -> int:
		low, high = 0, k * 100
		
		while low <= high:
			mid = (low + high) // 2
			total_price = self.calculate_total_price(mid, x)
			if total_price <= k:
				low = mid + 1
			else:
				high = mid - 1
				
		return high
		
	def calculate_total_price(self, num, x):
		if num == 0:
			return 0
		total = 0
		j = x
		while True:
			half = 1 << (j - 1)
			if half > num + 1:
				break
			base = half << 1
			quotient = (num + 1) // base
			remainder = (num + 1) % base
			count_j = quotient * half + max(0, remainder - half)
			total += count_j
			j += x
		return total
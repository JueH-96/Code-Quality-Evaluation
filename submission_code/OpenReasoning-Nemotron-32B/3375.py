import math
from math import gcd
from typing import List

class Solution:
	def findKthSmallest(self, coins: List[int], k: int) -> int:
		min_coin = min(coins)
		
		def lcm(a, b):
			return a * b // gcd(a, b)
		
		def count_le(x):
			total = 0
			n = len(coins)
			for bitmask in range(1, 1 << n):
				current_lcm = 1
				count_bits = 0
				for j in range(n):
					if bitmask & (1 << j):
						count_bits += 1
						current_lcm = lcm(current_lcm, coins[j])
						if current_lcm > x:
							break
				else:
					if count_bits % 2 == 1:
						total += x // current_lcm
					else:
						total -= x // current_lcm
			return total
		
		low, high = 0, k * min_coin
		while low < high:
			mid = (low + high) // 2
			if count_le(mid) < k:
				low = mid + 1
			else:
				high = mid
		return low
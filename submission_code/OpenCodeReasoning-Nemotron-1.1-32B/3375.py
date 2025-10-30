import math
from typing import List

class Solution:
	def findKthSmallest(self, coins: List[int], k: int) -> int:
		n = len(coins)
		low = min(coins)
		high = k * low
		
		def count_up_to(x):
			total = 0
			for mask in range(1, 1 << n):
				L = 1
				count_bits = 0
				for j in range(n):
					if mask & (1 << j):
						count_bits += 1
						g = math.gcd(L, coins[j])
						new_L = L * coins[j] // g
						if new_L > x:
							L = new_L
							break
						else:
							L = new_L
				if L > x:
					continue
				if count_bits % 2 == 1:
					total += x // L
				else:
					total -= x // L
			return total
		
		while low < high:
			mid = (low + high) // 2
			if count_up_to(mid) >= k:
				high = mid
			else:
				low = mid + 1
				
		return low
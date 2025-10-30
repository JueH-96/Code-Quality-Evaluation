from typing import List

class Solution:
	def findPrimePairs(self, n: int) -> List[List[int]]:
		if n < 2:
			return []
		
		is_prime = [True] * (n + 1)
		is_prime[0] = False
		is_prime[1] = False
		
		if n >= 4:
			for j in range(4, n + 1, 2):
				is_prime[j] = False
		
		sqrt_n = int(n ** 0.5)
		for i in range(3, sqrt_n + 1, 2):
			if is_prime[i]:
				start = i * i
				step = 2 * i
				if start > n:
					continue
				for j in range(start, n + 1, step):
					is_prime[j] = False
		
		res = []
		for x in range(1, n // 2 + 1):
			y = n - x
			if is_prime[x] and is_prime[y]:
				res.append([x, y])
		return res
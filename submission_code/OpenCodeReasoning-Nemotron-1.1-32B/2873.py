class Solution:
	def findPrimePairs(self, n: int) -> List[List[int]]:
		if n < 2:
			return []
		sieve = [True] * (n + 1)
		sieve[0] = False
		sieve[1] = False
		sqrt_n = int(n ** 0.5)
		for i in range(2, sqrt_n + 1):
			if sieve[i]:
				for j in range(i * i, n + 1, i):
					sieve[j] = False
		
		res = []
		for x in range(2, n // 2 + 1):
			if sieve[x] and sieve[n - x]:
				res.append([x, n - x])
		return res
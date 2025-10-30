class Solution:
	def valueAfterKSeconds(self, n: int, k: int) -> int:
		mod = 10**9 + 7
		total = n - 1 + k
		r = min(k, n - 1)
		
		if r == 0:
			return 1
		
		res = 1
		for i in range(1, r + 1):
			res = (res * (total - i + 1)) % mod
			res = (res * pow(i, mod - 2, mod)) % mod
		
		return res
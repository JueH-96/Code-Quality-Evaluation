class Solution:
	def minBitwiseArray(self, nums: List[int]) -> List[int]:
		res = []
		for p in nums:
			L = 0
			while p & (1 << L):
				L += 1
			if L == 0:
				res.append(-1)
			else:
				modulus = 1 << (L + 1)
				remainder = (1 << L) - 1
				a = (p - remainder) // modulus
				x = a * modulus + (1 << (L - 1)) - 1
				res.append(x)
		return res
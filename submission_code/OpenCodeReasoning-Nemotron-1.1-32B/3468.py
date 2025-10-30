class Solution:
	def getEncryptedString(self, s: str, k: int) -> str:
		n = len(s)
		k_mod = k % n
		res = []
		for i in range(n):
			j = (i + k_mod) % n
			res.append(s[j])
		return ''.join(res)
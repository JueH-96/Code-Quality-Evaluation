class Solution:
	def getEncryptedString(self, s: str, k: int) -> str:
		n = len(s)
		if n == 0:
			return ""
		k_mod = k % n
		res = []
		for i in range(n):
			idx = (i + k_mod) % n
			res.append(s[idx])
		return ''.join(res)
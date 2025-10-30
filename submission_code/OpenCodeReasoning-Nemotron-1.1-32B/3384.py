class Solution:
	def minimumOperationsToMakeKPeriodic(self, word: str, k: int) -> int:
		mod1 = 10**9 + 7
		mod2 = 10**9 + 9
		base1 = 131
		base2 = 1331
		n = len(word)
		h1 = [0] * (n + 1)
		h2 = [0] * (n + 1)
		p1 = [1] * (n + 1)
		p2 = [1] * (n + 1)
		
		for i in range(1, n + 1):
			c = word[i - 1]
			val = ord(c) - ord('a') + 1
			h1[i] = (h1[i - 1] * base1 + val) % mod1
			h2[i] = (h2[i - 1] * base2 + val) % mod2
			p1[i] = (p1[i - 1] * base1) % mod1
			p2[i] = (p2[i - 1] * base2) % mod2
		
		m = n // k
		cnt = {}
		for i in range(0, n, k):
			H1 = (h1[i + k] - h1[i] * p1[k]) % mod1
			H2 = (h2[i + k] - h2[i] * p2[k]) % mod2
			key = (H1, H2)
			cnt[key] = cnt.get(key, 0) + 1
		
		max_freq = max(cnt.values()) if cnt else 0
		return m - max_freq
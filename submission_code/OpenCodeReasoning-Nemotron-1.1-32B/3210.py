class Solution:
	def beautifulSubstrings(self, s: str, k: int) -> int:
		vowels_set = set('aeiou')
		n = len(s)
		pref = [0] * (n + 1)
		for i in range(n):
			if s[i] in vowels_set:
				pref[i + 1] = pref[i] + 1
			else:
				pref[i + 1] = pref[i]
		
		total = 0
		for L in range(2, n + 1, 2):
			x = L // 2
			if (x * x) % k != 0:
				continue
			for i in range(0, n - L + 1):
				j = i + L
				vowels_count = pref[j] - pref[i]
				if vowels_count == x:
					total += 1
		return total
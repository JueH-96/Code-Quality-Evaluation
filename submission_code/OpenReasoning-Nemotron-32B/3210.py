class Solution:
	def beautifulSubstrings(self, s: str, k: int) -> int:
		n = len(s)
		pref = [0] * (n + 1)
		vowels = "aeiou"
		for i in range(n):
			if s[i] in vowels:
				pref[i+1] = pref[i] + 1
			else:
				pref[i+1] = pref[i]
		
		valid_lengths = []
		for L in range(2, n + 1, 2):
			half = L // 2
			if (half * half) % k == 0:
				valid_lengths.append(L)
				
		total = 0
		for L in valid_lengths:
			target = L // 2
			for start in range(0, n - L + 1):
				end = start + L
				if pref[end] - pref[start] == target:
					total += 1
		return total
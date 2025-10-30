class Solution:
	def validSubstringCount(self, word1: str, word2: str) -> int:
		n = len(word2)
		L = len(word1)
		if n > L:
			return 0
		
		freq2 = [0] * 26
		for c in word2:
			idx = ord(c) - ord('a')
			freq2[idx] += 1
		
		total_required = 0
		for count in freq2:
			if count > 0:
				total_required += 1
		
		freq = [0] * 26
		formed = 0
		j = 0
		ans = 0
		for i in range(L):
			while j < L and (j - i < n or formed < total_required):
				c = word1[j]
				idx = ord(c) - ord('a')
				freq[idx] += 1
				if freq2[idx] > 0:
					if freq[idx] == freq2[idx]:
						formed += 1
				j += 1
			
			if formed == total_required and j - i >= n:
				ans += L - j + 1
			
			c = word1[i]
			idx = ord(c) - ord('a')
			if freq2[idx] > 0:
				if freq[idx] == freq2[idx]:
					formed -= 1
			freq[idx] -= 1
		
		return ans
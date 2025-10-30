class Solution:
	def validSubstringCount(self, word1: str, word2: str) -> int:
		n = len(word1)
		m = len(word2)
		required = [0] * 26
		for c in word2:
			idx = ord(c) - ord('a')
			required[idx] += 1
		
		total_required = 0
		for count in required:
			if count > 0:
				total_required += 1
		
		l = 0
		r = 0
		formed = 0
		freq = [0] * 26
		ans = 0
		
		while l < n:
			while r < n and formed < total_required:
				c = word1[r]
				idx = ord(c) - ord('a')
				if required[idx] > 0:
					freq[idx] += 1
					if freq[idx] == required[idx]:
						formed += 1
				r += 1
			
			if formed == total_required:
				ans += n - r + 1
			
			c = word1[l]
			idx = ord(c) - ord('a')
			if required[idx] > 0:
				if freq[idx] == required[idx]:
					formed -= 1
				freq[idx] -= 1
			l += 1
		
		return ans
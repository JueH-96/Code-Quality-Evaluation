class Solution:
	def longestSemiRepetitiveSubstring(self, s: str) -> int:
		n = len(s)
		if n == 0:
			return 0
		left = 0
		count = 0
		max_len = 1
		
		for right in range(1, n):
			if s[right] == s[right-1]:
				count += 1
				
			while count > 1:
				if left < n - 1 and s[left] == s[left+1]:
					count -= 1
				left += 1
				
			max_len = max(max_len, right - left + 1)
			
		return max_len
class Solution:
	def hasMatch(self, s: str, p: str) -> bool:
		parts = p.split('*')
		prefix = parts[0]
		suffix = parts[1]
		n = len(s)
		m = len(prefix)
		k = len(suffix)
		
		if m == 0 and k == 0:
			return True
			
		if m + k > n:
			return False
			
		for i in range(0, n - m + 1):
			if s[i:i+m] != prefix:
				continue
			if k == 0:
				return True
			for j in range(i + m, n - k + 1):
				if s[j:j+k] == suffix:
					return True
		return False
class Solution:
	def hasMatch(self, s: str, p: str) -> bool:
		idx = p.index('*')
		prefix = p[:idx]
		suffix = p[idx+1:]
		n = len(s)
		
		if not prefix and not suffix:
			return True
			
		if not prefix:
			return suffix in s
			
		if not suffix:
			return prefix in s
			
		len_pre = len(prefix)
		for i in range(n - len_pre + 1):
			if s[i:i+len_pre] == prefix:
				if s.find(suffix, i + len_pre) != -1:
					return True
		return False
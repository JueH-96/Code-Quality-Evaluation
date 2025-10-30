class Solution:
	def isSubstringPresent(self, s: str) -> bool:
		rev = s[::-1]
		rev_subs = {rev[i:i+2] for i in range(len(rev) - 1)}
		for i in range(len(s) - 1):
			if s[i:i+2] in rev_subs:
				return True
		return False
class Solution:
	def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
		if s1[0] != s2[0] or s1[0] != s3[0]:
			return -1
		
		n = min(len(s1), len(s2), len(s3))
		L = 0
		for i in range(n):
			if s1[i] == s2[i] == s3[i]:
				L = i + 1
			else:
				break
				
		return (len(s1) - L) + (len(s2) - L) + (len(s3) - L)
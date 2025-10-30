class Solution:
	def minimumOperations(self, num: str) -> int:
		n = len(num)
		candidate = n  # Worst case: remove all digits to form 0
		
		if '0' in num:
			candidate = min(candidate, n - 1)
		
		patterns = [("0", "0"), ("2", "5"), ("5", "0"), ("7", "5")]
		
		for p in patterns:
			best_i = -1
			for j in range(n):
				if num[j] == p[1] and best_i != -1:
					candidate = min(candidate, n - (best_i + 2))
				if num[j] == p[0]:
					best_i = j
					
		return candidate
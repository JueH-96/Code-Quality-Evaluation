class Solution:
	def minOperations(self, k: int) -> int:
		ans = float('inf')
		n = 1
		while n * n <= k:
			x = (k + n - 1) // n
			total_ops = n + x - 2
			if total_ops < ans:
				ans = total_ops
				
			m = k // n
			x2 = (k + m - 1) // m
			total_ops2 = m + x2 - 2
			if total_ops2 < ans:
				ans = total_ops2
				
			m1 = m + 1
			x3 = (k + m1 - 1) // m1
			total_ops3 = m1 + x3 - 2
			if total_ops3 < ans:
				ans = total_ops3
				
			n += 1
			
		return ans
class Solution:
	def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
		n = len(usageLimits)
		A = sorted(usageLimits)
		P = [0] * (n + 1)
		for i in range(1, n + 1):
			P[i] = P[i - 1] + A[i - 1]
		
		idx = 0
		j = 1
		while True:
			while idx < n and A[idx] < j:
				idx += 1
			count_ge = n - idx
			sum_less = P[idx]
			total_available = j * count_ge + sum_less
			required = j * (j + 1) // 2
			if total_available < required:
				break
			j += 1
		return j - 1
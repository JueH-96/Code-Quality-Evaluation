import bisect

class Solution:
	def minimumCost(self, nums: List[int]) -> int:
		n = len(nums)
		A = sorted(nums)
		P = [0] * (n + 1)
		for i in range(1, n + 1):
			P[i] = P[i - 1] + A[i - 1]
		
		palindromes = []
		for d in range(1, 10):
			start = 10**((d - 1) // 2)
			end = 10**((d + 1) // 2) - 1
			for num in range(start, end + 1):
				s = str(num)
				if d % 2 == 0:
					candidate = s + s[::-1]
				else:
					candidate = s + s[:-1][::-1]
				p_val = int(candidate)
				if p_val < 10**9:
					palindromes.append(p_val)
		
		min_cost = float('inf')
		for y in palindromes:
			idx = bisect.bisect_right(A, y)
			left_sum = y * idx - P[idx]
			right_sum = (P[n] - P[idx]) - y * (n - idx)
			total_cost = left_sum + right_sum
			if total_cost < min_cost:
				min_cost = total_cost
		return min_cost
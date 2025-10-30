class Solution:
	def maxSum(self, nums: List[int], k: int) -> int:
		MOD = 10**9 + 7
		max_bit = 60
		count = [0] * (max_bit + 1)
		
		for num in nums:
			for j in range(max_bit + 1):
				if num & (1 << j):
					count[j] += 1
		
		A = [0] * k
		for j in range(max_bit, -1, -1):
			num_assign = min(count[j], k)
			for i in range(num_assign):
				A[i] += (1 << j)
		
		ans = 0
		for num in A:
			ans = (ans + num * num) % MOD
		
		return ans
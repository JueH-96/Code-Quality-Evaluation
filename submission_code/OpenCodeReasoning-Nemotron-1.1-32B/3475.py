class Solution:
	def minOperations(self, nums: List[int]) -> int:
		n = len(nums)
		f = [0] * n
		
		for i in range(n - 2):
			base = nums[i]
			if i - 2 >= 0:
				base ^= f[i - 2]
			if i - 1 >= 0:
				base ^= f[i - 1]
			f[i] = 1 ^ base
		
		base_n2 = nums[n - 2]
		if n - 4 >= 0:
			base_n2 ^= f[n - 4]
		if n - 3 >= 0:
			base_n2 ^= f[n - 3]
		
		base_n1 = nums[n - 1]
		if n - 3 >= 0:
			base_n1 ^= f[n - 3]
		
		if base_n2 == 1 and base_n1 == 1:
			return sum(f[:n - 2])
		else:
			return -1
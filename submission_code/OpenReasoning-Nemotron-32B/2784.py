class Solution:
	def sumOfPower(self, nums: List[int]) -> int:
		mod = 10**9 + 7
		nums.sort()
		n = len(nums)
		total = 0
		for x in nums:
			total = (total + x * x % mod * x) % mod
		
		F_val = 0
		for j in range(1, n):
			F_val = (2 * F_val + nums[j-1]) % mod
			total = (total + (nums[j] * nums[j] % mod) * F_val) % mod
		
		return total
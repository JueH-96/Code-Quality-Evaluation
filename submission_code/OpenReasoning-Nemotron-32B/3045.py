class Solution:
	def minimumRightShifts(self, nums: List[int]) -> int:
		n = len(nums)
		min_val = min(nums)
		min_index = nums.index(min_val)
		candidate = nums[min_index:] + nums[:min_index]
		for i in range(len(candidate) - 1):
			if candidate[i] > candidate[i+1]:
				return -1
		return (n - min_index) % n
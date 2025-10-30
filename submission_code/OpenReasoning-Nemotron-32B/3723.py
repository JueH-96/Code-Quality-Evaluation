class Solution:
	def sumOfGoodNumbers(self, nums: List[int], k: int) -> int:
		total = 0
		n = len(nums)
		for i in range(n):
			left_ok = (i - k < 0) or (nums[i] > nums[i - k])
			right_ok = (i + k >= n) or (nums[i] > nums[i + k])
			if left_ok and right_ok:
				total += nums[i]
		return total
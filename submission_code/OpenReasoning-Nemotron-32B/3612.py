class Solution:
	def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
		n = len(nums)
		for i in range(n - 2 * k + 1):
			if all(nums[i + j] < nums[i + j + 1] for j in range(k - 1)):
				if all(nums[i + k + j] < nums[i + k + j + 1] for j in range(k - 1)):
					return True
		return False
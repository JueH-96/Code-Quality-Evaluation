class Solution:
	def minOperations(self, nums: List[int], k: int) -> int:
		collected = set()
		operations = 0
		for i in range(len(nums)-1, -1, -1):
			operations += 1
			if nums[i] <= k:
				collected.add(nums[i])
			if len(collected) == k:
				return operations
		return len(nums)
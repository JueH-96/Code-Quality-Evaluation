class Solution:
	def minOperations(self, nums: List[int], k: int) -> int:
		found = [False] * (k + 1)
		count = 0
		operations = 0
		for i in range(len(nums) - 1, -1, -1):
			operations += 1
			num = nums[i]
			if num <= k and not found[num]:
				found[num] = True
				count += 1
				if count == k:
					break
		return operations
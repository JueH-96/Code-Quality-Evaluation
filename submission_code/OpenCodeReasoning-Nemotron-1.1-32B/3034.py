class Solution:
	def numberOfPoints(self, nums: List[List[int]]) -> int:
		nums.sort(key=lambda x: x[0])
		start, end = nums[0]
		total = 0
		for i in range(1, len(nums)):
			if nums[i][0] <= end:
				end = max(end, nums[i][1])
			else:
				total += end - start + 1
				start, end = nums[i]
		total += end - start + 1
		return total
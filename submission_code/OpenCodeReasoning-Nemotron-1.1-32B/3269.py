class Solution:
	def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
		n = len(nums)
		m = len(pattern)
		count = 0
		for i in range(n - m):
			valid = True
			for k in range(m):
				if pattern[k] == 1:
					if nums[i + k + 1] <= nums[i + k]:
						valid = False
						break
				elif pattern[k] == 0:
					if nums[i + k + 1] != nums[i + k]:
						valid = False
						break
				else:
					if nums[i + k + 1] >= nums[i + k]:
						valid = False
						break
			if valid:
				count += 1
		return count
class Solution:
	def minimumRightShifts(self, nums: List[int]) -> int:
		n = len(nums)
		count = 0
		last_drop = -1
		for i in range(n):
			if nums[i] > nums[(i + 1) % n]:
				count += 1
				last_drop = i
		if count == 0:
			return 0
		elif count == 1:
			return n - (last_drop + 1)
		else:
			return -1
class Solution:
	def countValidSelections(self, nums: List[int]) -> int:
		n = len(nums)
		zeros = [i for i in range(n) if nums[i] == 0]
		count = 0
		for start in zeros:
			for direction in [1, -1]:
				arr = nums.copy()
				curr = start
				dir = direction
				while 0 <= curr < n:
					if arr[curr] == 0:
						curr += dir
					else:
						arr[curr] -= 1
						dir = -dir
						curr += dir
				if all(x == 0 for x in arr):
					count += 1
		return count
class Solution:
	def matrixSum(self, nums: List[List[int]]) -> int:
		for row in nums:
			row.sort(reverse=True)
		
		n = len(nums)
		pointers = [0] * n
		score = 0
		
		while True:
			current_max = 0
			found = False
			for i in range(n):
				if pointers[i] < len(nums[i]):
					found = True
					if nums[i][pointers[i]] > current_max:
						current_max = nums[i][pointers[i]]
					pointers[i] += 1
			if not found:
				break
			score += current_max
			
		return score
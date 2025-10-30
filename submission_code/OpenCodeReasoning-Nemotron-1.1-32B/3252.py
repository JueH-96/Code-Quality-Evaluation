class Solution:
	def incremovableSubarrayCount(self, nums: List[int]) -> int:
		n = len(nums)
		prefix = [True] * n
		for i in range(1, n):
			prefix[i] = prefix[i-1] and (nums[i] > nums[i-1])
		
		suffix = [True] * n
		for i in range(n-2, -1, -1):
			suffix[i] = suffix[i+1] and (nums[i] < nums[i+1])
		
		count = 0
		for l in range(n):
			for r in range(l, n):
				if l > 0 and not prefix[l-1]:
					continue
				if r < n-1 and not suffix[r+1]:
					continue
				if l > 0 and r < n-1:
					if nums[l-1] < nums[r+1]:
						count += 1
				else:
					count += 1
		return count
class Solution:
	def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
		n = len(nums)
		diff = [0] * (n + 2)
		for l, r in queries:
			diff[l] += 1
			diff[r + 1] -= 1
		
		count_arr = [0] * n
		cur = 0
		for i in range(n):
			cur += diff[i]
			count_arr[i] = cur
		
		for i in range(n):
			if nums[i] > count_arr[i]:
				return False
		return True
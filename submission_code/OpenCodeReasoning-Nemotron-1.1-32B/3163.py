class Solution:
	def sumCounts(self, nums: List[int]) -> int:
		n = len(nums)
		total = 0
		for i in range(n):
			freq = [0] * 101
			distinct = 0
			for j in range(i, n):
				num = nums[j]
				if freq[num] == 0:
					distinct += 1
				freq[num] += 1
				total += distinct * distinct
		return total
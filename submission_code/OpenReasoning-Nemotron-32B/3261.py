class Solution:
	def minOrAfterOperations(self, nums: List[int], k: int) -> int:
		n = len(nums)
		ans = 0
		for i in range(30):
			count_ones = 0
			for num in nums:
				if (num >> i) & 1:
					count_ones += 1
			if count_ones > k:
				ans |= (1 << i)
		return ans
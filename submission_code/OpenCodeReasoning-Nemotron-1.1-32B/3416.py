from typing import List

class Solution:
	def sumDigitDifferences(self, nums: List[int]) -> int:
		n = len(nums)
		total_pairs = n * (n - 1) // 2
		str_nums = [str(num) for num in nums]
		m = len(str_nums[0])
		
		total_diff = 0
		for j in range(m):
			freq = [0] * 10
			for i in range(n):
				digit = int(str_nums[i][j])
				freq[digit] += 1
			same = 0
			for cnt in freq:
				same += cnt * (cnt - 1) // 2
			total_diff += total_pairs - same
		
		return total_diff
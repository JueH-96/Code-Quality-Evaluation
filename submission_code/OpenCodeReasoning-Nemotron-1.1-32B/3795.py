from typing import List

class Solution:
	def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
		n = len(nums)
		dp_list = []
		for num in nums:
			dp = [False] * (num + 1)
			dp[0] = True
			dp_list.append(dp)
		
		if all(dp_list[j][nums[j]] for j in range(n)):
			return 0
		
		m = len(queries)
		for i in range(m):
			l, r, val = queries[i]
			for j in range(l, r + 1):
				if dp_list[j][nums[j]]:
					continue
				if val > nums[j]:
					continue
				for x in range(nums[j], val - 1, -1):
					if dp_list[j][x - val]:
						dp_list[j][x] = True
						if x == nums[j]:
							break
			if all(dp_list[j][nums[j]] for j in range(n)):
				return i + 1
		
		return -1
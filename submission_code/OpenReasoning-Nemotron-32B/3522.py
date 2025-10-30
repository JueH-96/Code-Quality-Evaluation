from typing import List

class Solution:
	def resultsArray(self, nums: List[int], k: int) -> List[int]:
		n = len(nums)
		if k == 1:
			return nums[:]
		
		arr = [0] * (n - 1)
		for i in range(n - 1):
			if nums[i+1] - nums[i] == 1:
				arr[i] = 1
		
		prefix = [0] * n
		for i in range(1, n):
			prefix[i] = prefix[i-1] + arr[i-1]
		
		res = []
		for i in range(n - k + 1):
			total = prefix[i + k - 1] - prefix[i]
			if total == k - 1:
				res.append(nums[i + k - 1])
			else:
				res.append(-1)
				
		return res
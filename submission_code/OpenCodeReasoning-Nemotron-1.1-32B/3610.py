from typing import List

class Solution:
	def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
		n = len(nums)
		ans = []
		for i in range(n - k + 1):
			total = 0
			freq = {}
			for j in range(i, i + k):
				num = nums[j]
				total += num
				freq[num] = freq.get(num, 0) + 1
			
			distinct = list(freq.keys())
			distinct.sort(key=lambda num: (-freq[num], -num))
			
			take = min(x, len(distinct))
			non_top_sum = 0
			for idx in range(take, len(distinct)):
				num = distinct[idx]
				non_top_sum += num * freq[num]
			
			ans.append(total - non_top_sum)
		
		return ans
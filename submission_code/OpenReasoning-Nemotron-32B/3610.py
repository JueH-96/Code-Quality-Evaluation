from typing import List

class Solution:
	def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
		n = len(nums)
		result = []
		for i in range(n - k + 1):
			sub = nums[i:i+k]
			freq_map = {}
			for num in sub:
				freq_map[num] = freq_map.get(num, 0) + 1
			
			items = [(count, num) for num, count in freq_map.items()]
			items.sort(key=lambda z: (-z[0], -z[1]))
			
			total = 0
			for j in range(min(x, len(items))):
				count, num = items[j]
				total += count * num
			result.append(total)
		return result
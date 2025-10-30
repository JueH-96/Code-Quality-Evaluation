from typing import List

class Solution:
	def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
		count = {}
		for num in nums:
			count[num] = count.get(num, 0) + 1
		
		for i in range(len(moveFrom)):
			f = moveFrom[i]
			t = moveTo[i]
			if f == t:
				continue
			c = count.pop(f)
			count[t] = count.get(t, 0) + c
		
		return sorted(count.keys())
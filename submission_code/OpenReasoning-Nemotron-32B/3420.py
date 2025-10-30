from typing import List

class Solution:
	def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
		occ = [i for i, num in enumerate(nums) if num == x]
		count = len(occ)
		return [occ[q-1] if q <= count else -1 for q in queries]
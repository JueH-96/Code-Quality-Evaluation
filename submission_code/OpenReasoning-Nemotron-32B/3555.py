class Solution:
	def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
		arr = nums.copy()
		for _ in range(k):
			m = min(arr)
			idx = arr.index(m)
			arr[idx] = m * multiplier
		return arr
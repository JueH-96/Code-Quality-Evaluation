class Solution:
	def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
		total_apples = sum(apple)
		capacity.sort(reverse=True)
		current = 0
		for i, cap in enumerate(capacity, 1):
			current += cap
			if current >= total_apples:
				return i
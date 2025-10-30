class Solution:
	def numberOfChild(self, n: int, k: int) -> int:
		current = 0
		direction = 1
		for _ in range(k):
			current += direction
			if current == 0 or current == n - 1:
				direction = -direction
		return current
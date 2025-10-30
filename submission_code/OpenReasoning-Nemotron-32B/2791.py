class Solution:
	def circularGameLosers(self, n: int, k: int) -> List[int]:
		visited = set([0])
		current = 0
		turn = 1
		while True:
			current = (current + turn * k) % n
			if current in visited:
				break
			visited.add(current)
			turn += 1
		return [i + 1 for i in range(n) if i not in visited]
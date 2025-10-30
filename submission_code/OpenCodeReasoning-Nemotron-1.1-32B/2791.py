class Solution:
	def circularGameLosers(self, n: int, k: int) -> List[int]:
		visited = [False] * n
		visited[0] = True
		current = 0
		i = 1
		while True:
			next_pos = (current + i * k) % n
			if visited[next_pos]:
				break
			visited[next_pos] = True
			current = next_pos
			i += 1
		
		result = []
		for j in range(n):
			if not visited[j]:
				result.append(j + 1)
		return result
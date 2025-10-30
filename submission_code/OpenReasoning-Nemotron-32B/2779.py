from typing import List

class Solution:
	def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
		colors = [0] * n
		total = 0
		res = []
		for idx, color in queries:
			if colors[idx] == color:
				res.append(total)
				continue
			for neighbor in [idx - 1, idx + 1]:
				if 0 <= neighbor < n:
					if colors[idx] != 0 and colors[neighbor] != 0 and colors[idx] == colors[neighbor]:
						total -= 1
			colors[idx] = color
			for neighbor in [idx - 1, idx + 1]:
				if 0 <= neighbor < n:
					if colors[idx] != 0 and colors[neighbor] != 0 and colors[idx] == colors[neighbor]:
						total += 1
			res.append(total)
		return res
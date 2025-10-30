from typing import List

class Solution:
	def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
		counts = [dict() for _ in range(n)]
		for move in pick:
			player = move[0]
			color = move[1]
			counts[player][color] = counts[player].get(color, 0) + 1
		
		win_count = 0
		for i in range(n):
			for color, cnt in counts[i].items():
				if cnt >= i + 1:
					win_count += 1
					break
		return win_count
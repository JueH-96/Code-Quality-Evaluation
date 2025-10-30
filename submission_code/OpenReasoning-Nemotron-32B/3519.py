from typing import List

class Solution:
	def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
		player_freq = [dict() for _ in range(n)]
		for x, y in pick:
			player_freq[x][y] = player_freq[x].get(y, 0) + 1
		
		count = 0
		for i in range(n):
			max_freq = max(player_freq[i].values()) if player_freq[i] else 0
			if max_freq >= i + 1:
				count += 1
		return count
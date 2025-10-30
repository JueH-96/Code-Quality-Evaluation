class Solution:
	def canAliceWin(self, n: int) -> bool:
		player = 0  # 0 for Alice, 1 for Bob
		stones = n
		move = 10
		
		while move > 0 and stones >= move:
			stones -= move
			move -= 1
			player = 1 - player  # Switch player
		
		return player == 1
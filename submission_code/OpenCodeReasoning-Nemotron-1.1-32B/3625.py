class Solution:
	def canAliceWin(self, n: int) -> bool:
		moves = 10
		turn = 0  # 0 for Alice, 1 for Bob
		stones = n
		while moves > 0 and stones >= moves:
			stones -= moves
			moves -= 1
			turn = 1 - turn
		return turn == 1
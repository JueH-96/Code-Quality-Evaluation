class Solution:
	def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
		row, col = 0, 0
		for cmd in commands:
			if cmd == "UP":
				row -= 1
			elif cmd == "RIGHT":
				col += 1
			elif cmd == "DOWN":
				row += 1
			elif cmd == "LEFT":
				col -= 1
		return row * n + col
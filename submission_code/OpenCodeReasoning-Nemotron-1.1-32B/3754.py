class Solution:
	def maxDistance(self, s: str, k: int) -> int:
		x, y = 0, 0
		changes = k
		best = 0
		for c in s:
			new_x, new_y = x, y
			if c == 'N':
				new_y += 1
			elif c == 'S':
				new_y -= 1
			elif c == 'E':
				new_x += 1
			elif c == 'W':
				new_x -= 1
			new_manhattan = abs(new_x) + abs(new_y)
			current_manhattan = abs(x) + abs(y)
			if new_manhattan > current_manhattan:
				x, y = new_x, new_y
			else:
				if changes > 0:
					best_move = None
					best_manhattan = current_manhattan
					for move in ['N', 'S', 'E', 'W']:
						nx, ny = x, y
						if move == 'N':
							ny += 1
						elif move == 'S':
							ny -= 1
						elif move == 'E':
							nx += 1
						elif move == 'W':
							nx -= 1
						nm = abs(nx) + abs(ny)
						if nm > best_manhattan:
							best_manhattan = nm
							best_move = move
					if best_move == 'N':
						y += 1
					elif best_move == 'S':
						y -= 1
					elif best_move == 'E':
						x += 1
					elif best_move == 'W':
						x -= 1
					changes -= 1
				else:
					x, y = new_x, new_y
			current_manhattan = abs(x) + abs(y)
			if current_manhattan > best:
				best = current_manhattan
		return best
import itertools

def rotate90(grid):
	reversed_grid = list(reversed(grid))
	rotated = list(zip(*reversed_grid))
	return [''.join(row) for row in rotated]

def normalize(poly):
	points = []
	for i in range(4):
		for j in range(4):
			if poly[i][j] == '#':
				points.append((i, j))
	if not points:
		return set(), 0, 0
	min_i = min(i for i, j in points)
	min_j = min(j for i, j in points)
	normalized_set = set()
	max_dx = 0
	max_dy = 0
	for (i, j) in points:
		dx = i - min_i
		dy = j - min_j
		normalized_set.add((dx, dy))
		if dx > max_dx:
			max_dx = dx
		if dy > max_dy:
			max_dy = dy
	return normalized_set, max_dx, max_dy

def main():
	lines = [input().strip() for _ in range(12)]
	polyominoes = []
	for i in range(3):
		polyominoes.append(lines[i*4:i*4+4])
	
	total_hashes = 0
	for poly in polyominoes:
		for line in poly:
			total_hashes += line.count('#')
	if total_hashes != 16:
		print("No")
		return

	poly_rots = []
	for poly in polyominoes:
		rotations = [poly]
		current = poly
		for _ in range(3):
			current = rotate90(current)
			rotations.append(current)
		rot_info = []
		for grid in rotations:
			s, dx, dy = normalize(grid)
			rot_info.append((s, dx, dy))
		poly_rots.append(rot_info)
	
	perms = list(itertools.permutations([0,1,2]))
	for perm in perms:
		for r0 in range(4):
			for r1 in range(4):
				for r2 in range(4):
					sets = []
					max_dx_list = []
					max_dy_list = []
					for i in range(3):
						poly_index = perm[i]
						rot_index = [r0, r1, r2][i]
						s, dx, dy = poly_rots[poly_index][rot_index]
						sets.append(s)
						max_dx_list.append(dx)
						max_dy_list.append(dy)
					
					grid = [[False] * 4 for _ in range(4)]
					
					for r0_pos in range(0, 4 - max_dx_list[0]):
						for c0_pos in range(0, 4 - max_dy_list[0]):
							valid0 = True
							for (dx, dy) in sets[0]:
								x = r0_pos + dx
								y = c0_pos + dy
								if grid[x][y]:
									valid0 = False
									break
							if not valid0:
								continue
							
							for (dx, dy) in sets[0]:
								x = r0_pos + dx
								y = c0_pos + dy
								grid[x][y] = True
							
							for r1_pos in range(0, 4 - max_dx_list[1]):
								for c1_pos in range(0, 4 - max_dy_list[1]):
									valid1 = True
									for (dx, dy) in sets[1]:
										x = r1_pos + dx
										y = c1_pos + dy
										if grid[x][y]:
											valid1 = False
											break
									if not valid1:
										continue
									
									for (dx, dy) in sets[1]:
										x = r1_pos + dx
										y = c1_pos + dy
										grid[x][y] = True
									
									for r2_pos in range(0, 4 - max_dx_list[2]):
										for c2_pos in range(0, 4 - max_dy_list[2]):
											valid2 = True
											for (dx, dy) in sets[2]:
												x = r2_pos + dx
												y = c2_pos + dy
												if grid[x][y]:
													valid2 = False
													break
											if not valid2:
												continue
											
											print("Yes")
											return
									
									for (dx, dy) in sets[1]:
										x = r1_pos + dx
										y = c1_pos + dy
										grid[x][y] = False
							
							for (dx, dy) in sets[0]:
								x = r0_pos + dx
								y = c0_pos + dy
								grid[x][y] = False
	print("No")

if __name__ == "__main__":
	main()
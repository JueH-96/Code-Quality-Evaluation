import sys

def generate_placements(grid):
	points = set()
	for i in range(4):
		for j in range(4):
			if grid[i][j] == '#':
				points.add((i, j))
	if not points:
		return set()
	
	rotations = []
	rotations.append(points)
	r1 = set()
	for (i, j) in points:
		r1.add((j, 3 - i))
	rotations.append(r1)
	r2 = set()
	for (i, j) in points:
		r2.add((3 - i, 3 - j))
	rotations.append(r2)
	r3 = set()
	for (i, j) in points:
		r3.add((3 - j, i))
	rotations.append(r3)
	
	placements_set = set()
	
	for rot in rotations:
		min_i = min(x[0] for x in rot)
		max_i = max(x[0] for x in rot)
		min_j = min(x[1] for x in rot)
		max_j = max(x[1] for x in rot)
		normalized = set()
		for (i, j) in rot:
			normalized.add((i - min_i, j - min_j))
		h = max_i - min_i + 1
		w = max_j - min_j + 1
		for dx in range(0, 4 - h + 1):
			for dy in range(0, 4 - w + 1):
				placement = set()
				for (i, j) in normalized:
					ni = i + dx
					nj = j + dy
					placement.add((ni, nj))
				placements_set.add(frozenset(placement))
				
	return placements_set

def main():
	data = sys.stdin.read().splitlines()
	if len(data) < 12:
		print("No")
		return
		
	polyominoes = []
	for i in range(3):
		p = data[i*4 : (i*4+4)]
		polyominoes.append(p)
	
	sizes = []
	for p in polyominoes:
		cnt = 0
		for row in p:
			cnt += row.count('#')
		sizes.append(cnt)
		
	if sum(sizes) != 16:
		print("No")
		return
		
	placements = []
	for p in polyominoes:
		placements.append(generate_placements(p))
		
	for a in placements[0]:
		for b in placements[1]:
			if a & b:
				continue
			for c in placements[2]:
				if a & c or b & c:
					continue
				if len(a) + len(b) + len(c) == 16:
					print("Yes")
					return
					
	print("No")

if __name__ == "__main__":
	main()
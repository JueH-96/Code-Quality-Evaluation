import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	H, W, D = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	floors = []
	for i in range(H):
		for j in range(W):
			if grid[i][j] == '.':
				floors.append((i, j))
				
	n = len(floors)
	cover_sets = []
	for cell in floors:
		cov_set = set()
		for other in floors:
			dist = abs(cell[0] - other[0]) + abs(cell[1] - other[1])
			if dist <= D:
				cov_set.add(other)
		cover_sets.append(cov_set)
	
	ans = 0
	for i in range(n):
		for j in range(i+1, n):
			union_set = cover_sets[i] | cover_sets[j]
			if len(union_set) > ans:
				ans = len(union_set)
				
	print(ans)

if __name__ == '__main__':
	main()
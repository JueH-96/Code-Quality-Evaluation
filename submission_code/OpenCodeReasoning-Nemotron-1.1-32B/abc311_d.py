from collections import deque

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, m = map(int, data[0].split())
	grid = []
	for i in range(1, 1 + n):
		grid.append(data[i].strip())
	
	visited = set()
	touched = set()
	queue = deque()
	start = (1, 1)
	visited.add(start)
	touched.add(start)
	queue.append(start)
	
	directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
	
	while queue:
		r, c = queue.popleft()
		for dr, dc in directions:
			nr = r + dr
			nc = c + dc
			if grid[nr][nc] == '#':
				continue
			cur_r, cur_c = nr, nc
			while True:
				touched.add((cur_r, cur_c))
				next_r = cur_r + dr
				next_c = cur_c + dc
				if grid[next_r][next_c] == '#':
					break
				else:
					cur_r, cur_c = next_r, next_c
			if (cur_r, cur_c) not in visited:
				visited.add((cur_r, cur_c))
				queue.append((cur_r, cur_c))
				
	print(len(touched))

if __name__ == '__main__':
	main()
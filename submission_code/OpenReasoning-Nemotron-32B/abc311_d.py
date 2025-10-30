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
	
	visited = [[False] * m for _ in range(n)]
	directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
	queue = deque()
	enqueued = set()
	
	if grid[1][1] == '.':
		visited[1][1] = True
		queue.append((1, 1))
		enqueued.add((1, 1))
	
	while queue:
		i, j = queue.popleft()
		for dx, dy in directions:
			cur_i, cur_j = i, j
			while True:
				next_i = cur_i + dx
				next_j = cur_j + dy
				if grid[next_i][next_j] == '#':
					stopping_cell = (cur_i, cur_j)
					break
				if not visited[next_i][next_j]:
					visited[next_i][next_j] = True
				cur_i, cur_j = next_i, next_j
			if stopping_cell not in enqueued:
				enqueued.add(stopping_cell)
				queue.append(stopping_cell)
	
	count = 0
	for i in range(n):
		for j in range(m):
			if visited[i][j]:
				count += 1
	print(count)

if __name__ == '__main__':
	main()
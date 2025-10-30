import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	H = int(data[0])
	W = int(data[1])
	X = int(data[2])
	P = int(data[3])
	Q = int(data[4])
	grid = []
	index = 5
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		grid.append(row)
		index += W
		
	P0 = P - 1
	Q0 = Q - 1
	
	total = grid[P0][Q0]
	visited = [[False] * W for _ in range(H)]
	heap = []
	visited[P0][Q0] = True
	
	directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
	
	for dx, dy in directions:
		ni = P0 + dx
		nj = Q0 + dy
		if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
			visited[ni][nj] = True
			heapq.heappush(heap, (grid[ni][nj], ni, nj))
			
	while heap:
		s, i, j = heapq.heappop(heap)
		if total > s * X:
			total += s
			for dx, dy in directions:
				ni = i + dx
				nj = j + dy
				if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj]:
					visited[ni][nj] = True
					heapq.heappush(heap, (grid[ni][nj], ni, nj))
		else:
			break
			
	print(total)

if __name__ == "__main__":
	main()
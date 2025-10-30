import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	H = int(data[0])
	W = int(data[1])
	Y = int(data[2])
	grid = []
	index = 3
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		grid.append(row)
		index += W
		
	INF = 10**10
	dist = [[INF] * W for _ in range(H)]
	heap = []
	dirs = [(0,1), (1,0), (0,-1), (-1,0)]
	
	for i in range(H):
		for j in range(W):
			if i == 0 or i == H-1 or j == 0 or j == W-1:
				dist[i][j] = grid[i][j]
				heapq.heappush(heap, (dist[i][j], i, j))
				
	while heap:
		d, i, j = heapq.heappop(heap)
		if d != dist[i][j]:
			continue
		for dx, dy in dirs:
			ni, nj = i + dx, j + dy
			if 0 <= ni < H and 0 <= nj < W:
				nd = max(d, grid[ni][nj])
				if nd < dist[ni][nj]:
					dist[ni][nj] = nd
					heapq.heappush(heap, (nd, ni, nj))
					
	freq = [0] * (Y + 1)
	for i in range(H):
		for j in range(W):
			t0 = dist[i][j]
			if t0 <= Y:
				freq[t0] += 1
				
	sunk_by = [0] * (Y + 1)
	for t in range(1, Y + 1):
		sunk_by[t] = sunk_by[t - 1] + freq[t]
		
	total_cells = H * W
	for t in range(1, Y + 1):
		print(total_cells - sunk_by[t])

if __name__ == "__main__":
	main()
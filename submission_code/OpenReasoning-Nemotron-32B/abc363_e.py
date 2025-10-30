import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	H = int(data[0])
	W = int(data[1])
	Y = int(data[2])
	grid = []
	index = 3
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		grid.append(row)
		index += W
		
	if H == 0 or W == 0:
		for _ in range(Y):
			print(0)
		return
			
	INF = 10**9
	d = [[INF] * W for _ in range(H)]
	heap = []
	dirs = [(0,1), (0,-1), (1,0), (-1,0)]
	
	for i in range(H):
		for j in range(W):
			if i == 0 or i == H-1 or j == 0 or j == W-1:
				d[i][j] = grid[i][j]
				heapq.heappush(heap, (d[i][j], i, j))
				
	while heap:
		val, i, j = heapq.heappop(heap)
		if val != d[i][j]:
			continue
		for dx, dy in dirs:
			ni, nj = i + dx, j + dy
			if 0 <= ni < H and 0 <= nj < W:
				new_val = max(val, grid[ni][nj])
				if new_val < d[ni][nj]:
					d[ni][nj] = new_val
					heapq.heappush(heap, (new_val, ni, nj))
					
	max_val = 100000
	freq = [0] * (max_val + 1)
	for i in range(H):
		for j in range(W):
			if d[i][j] <= max_val:
				freq[d[i][j]] += 1
				
	total = H * W
	cum = 0
	results = []
	for s in range(1, Y + 1):
		if s <= max_val:
			cum += freq[s]
		results.append(total - cum)
		
	for res in results:
		print(res)

if __name__ == '__main__':
	main()
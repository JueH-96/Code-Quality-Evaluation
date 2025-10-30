import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
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
		
	best = [[0] * W for _ in range(H)]
	start_i = P-1
	start_j = Q-1
	best[start_i][start_j] = grid[start_i][start_j]
	heap = []
	heapq.heappush(heap, (-best[start_i][start_j], start_i, start_j))
	
	while heap:
		neg_strength, i, j = heapq.heappop(heap)
		strength = -neg_strength
		if strength != best[i][j]:
			continue
		for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
			ni, nj = i + dx, j + dy
			if 0 <= ni < H and 0 <= nj < W:
				if grid[ni][nj] < strength / X:
					new_strength = strength + grid[ni][nj]
					if new_strength > best[ni][nj]:
						best[ni][nj] = new_strength
						heapq.heappush(heap, (-new_strength, ni, nj))
						
	ans = 0
	for i in range(H):
		for j in range(W):
			if best[i][j] > ans:
				ans = best[i][j]
				
	print(ans)

if __name__ == "__main__":
	main()
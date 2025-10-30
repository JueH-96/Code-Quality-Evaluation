import sys
from collections import deque
import heapq

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print("No")
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	N = int(data[1+H])
	medicines = []
	for i in range(1+H+1, 1+H+1+N):
		parts = data[i].split()
		if len(parts) < 3:
			continue
		r, c, e = map(int, parts)
		r -= 1
		c -= 1
		medicines.append((r, c, e))
	
	start = None
	goal = None
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 'S':
				start = (i, j)
			elif grid[i][j] == 'T':
				goal = (i, j)
				
	if start is None or goal is None:
		print("No")
		return
		
	points = [start, goal]
	for (r, c, e) in medicines:
		points.append((r, c))
	
	n_nodes = len(points)
	INF = 10**9
	D = [[INF] * n_nodes for _ in range(n_nodes)]
	
	dirs = [(1,0), (-1,0), (0,1), (0,-1)]
	
	for idx in range(n_nodes):
		r0, c0 = points[idx]
		dist_grid = [[-1] * W for _ in range(H)]
		q = deque()
		dist_grid[r0][c0] = 0
		q.append((r0, c0))
		while q:
			r, c = q.popleft()
			for dr, dc in dirs:
				nr, nc = r+dr, c+dc
				if 0 <= nr < H and 0 <= nc < W:
					if dist_grid[nr][nc] == -1 and grid[nr][nc] != '#':
						dist_grid[nr][nc] = dist_grid[r][c] + 1
						q.append((nr, nc))
		for j in range(n_nodes):
			rj, cj = points[j]
			d_val = dist_grid[rj][cj]
			if d_val == -1:
				D[idx][j] = INF
			else:
				D[idx][j] = d_val
				
	med_energy = [0] * n_nodes
	for idx in range(2, n_nodes):
		med_energy[idx] = medicines[idx-2][2]
	
	dp = [[-1] * 2 for _ in range(n_nodes)]
	heap = []
	dp[0][0] = 0
	heapq.heappush(heap, (0, 0, 0))

	while heap:
		neg_e, i, flag = heapq.heappop(heap)
		e = -neg_e

		if i == 1 and flag == 0:
			print("Yes")
			return

		if e < dp[i][flag]:
			continue

		if i >= 2 and flag == 0:
			new_energy = med_energy[i]
			if new_energy > dp[i][1]:
				dp[i][1] = new_energy
				heapq.heappush(heap, (-new_energy, i, 1))

		for j in range(n_nodes):
			if j == i:
				continue
			if e < D[i][j]:
				continue
			new_energy = e - D[i][j]
			if j == 0 or j == 1:
				if new_energy > dp[j][0]:
					dp[j][0] = new_energy
					heapq.heappush(heap, (-new_energy, j, 0))
			else:
				if new_energy > dp[j][0]:
					dp[j][0] = new_energy
					heapq.heappush(heap, (-new_energy, j, 0))

	print("No")

if __name__ == "__main__":
	main()
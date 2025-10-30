import heapq

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		print("No")
		return
	
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(list(data[i].strip()))
	
	s_pos = None
	t_pos = None
	for i in range(H):
		for j in range(W):
			if grid[i][j] == 'S':
				s_pos = (i, j)
				grid[i][j] = '.'
			elif grid[i][j] == 'T':
				t_pos = (i, j)
				grid[i][j] = '.'
				
	if s_pos is None or t_pos is None:
		print("No")
		return
		
	if s_pos == t_pos:
		print("Yes")
		return
		
	N = int(data[1+H].strip())
	med_dict = {}
	index = 1+H+1
	for i in range(N):
		parts = data[index].split()
		index += 1
		r = int(parts[0]) - 1
		c = int(parts[1]) - 1
		E = int(parts[2])
		med_dict[(r, c)] = E
		
	state0 = [[-1] * W for _ in range(H)]
	state1 = [[-1] * W for _ in range(H)]
	q = []
	
	sr, sc = s_pos
	if (sr, sc) in med_dict:
		state0[sr][sc] = 0
		state1[sr][sc] = med_dict[(sr, sc)]
		heapq.heappush(q, (0, sr, sc, 0))
		heapq.heappush(q, (-med_dict[(sr, sc)], sr, sc, 1))
	else:
		state0[sr][sc] = 0
		heapq.heappush(q, (0, sr, sc, 0))
		
	directions = [(0,1), (0,-1), (1,0), (-1,0)]
	
	while q:
		neg_energy, r, c, flag = heapq.heappop(q)
		energy = -neg_energy
		
		if flag == 0:
			if energy != state0[r][c]:
				continue
		else:
			if energy != state1[r][c]:
				continue
				
		if (r, c) == t_pos:
			print("Yes")
			return
			
		if flag == 0:
			if (r, c) in med_dict:
				E_med = med_dict[(r, c)]
				if E_med > state1[r][c]:
					state1[r][c] = E_med
					heapq.heappush(q, (-E_med, r, c, 1))
					
		for dr, dc in directions:
			nr = r + dr
			nc = c + dc
			if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
				if energy >= 1:
					new_energy = energy - 1
					if new_energy > state0[nr][nc]:
						state0[nr][nc] = new_energy
						heapq.heappush(q, (-new_energy, nr, nc, 0))
						
	print("No")

if __name__ == "__main__":
	main()
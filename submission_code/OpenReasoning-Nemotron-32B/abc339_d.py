from collections import deque

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	
	n = int(data[0].strip())
	grid = []
	for i in range(1, 1+n):
		grid.append(data[i].strip())
	
	players = []
	for i in range(n):
		for j in range(n):
			if grid[i][j] == 'P':
				players.append((i, j))
				
	if len(players) < 2:
		print(-1)
		return
		
	r1, c1 = players[0]
	r2, c2 = players[1]
	
	dist = [[[[-1] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
	moves = [(0,1), (0,-1), (1,0), (-1,0)]
	
	q = deque()
	dist[r1][c1][r2][c2] = 0
	q.append((r1, c1, r2, c2))
	
	while q:
		a, b, c, d = q.popleft()
		step = dist[a][b][c][d]
		
		if a == c and b == d:
			print(step)
			return
			
		for dx, dy in moves:
			na1 = a + dx
			nb1 = b + dy
			if na1 < 0 or na1 >= n or nb1 < 0 or nb1 >= n or grid[na1][nb1] == '#':
				na1, nb1 = a, b
				
			na2 = c + dx
			nb2 = d + dy
			if na2 < 0 or na2 >= n or nb2 < 0 or nb2 >= n or grid[na2][nb2] == '#':
				na2, nb2 = c, d
				
			if dist[na1][nb1][na2][nb2] == -1:
				dist[na1][nb1][na2][nb2] = step + 1
				q.append((na1, nb1, na2, nb2))
				
	print(-1)

if __name__ == "__main__":
	main()
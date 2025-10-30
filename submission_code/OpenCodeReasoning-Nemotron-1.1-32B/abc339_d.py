import collections

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
	N = int(data[0].strip())
	grid = []
	for i in range(1, 1+N):
		grid.append(data[i].strip())
	
	players = []
	for i in range(N):
		for j in range(N):
			if grid[i][j] == 'P':
				players.append((i, j))
				
	if len(players) != 2:
		print(-1)
		return
		
	A, B = players[0], players[1]
	if A <= B:
		start_state = (A[0], A[1], B[0], B[1])
	else:
		start_state = (B[0], B[1], A[0], A[1])
		
	visited = [[[[False] * N for _ in range(N)] for _ in range(N)] for _ in range(N)]
	directions = [(0,1), (0,-1), (1,0), (-1,0)]
	
	q = collections.deque()
	q.append((start_state[0], start_state[1], start_state[2], start_state[3], 0))
	visited[start_state[0]][start_state[1]][start_state[2]][start_state[3]] = True
	
	found = False
	while q:
		r1, c1, r2, c2, steps = q.popleft()
		
		if r1 == r2 and c1 == c2:
			print(steps)
			found = True
			break
			
		for dr, dc in directions:
			nr1, nc1 = r1 + dr, c1 + dc
			if not (0 <= nr1 < N and 0 <= nc1 < N) or grid[nr1][nc1] == '#':
				nr1, nc1 = r1, c1
				
			nr2, nc2 = r2 + dr, c2 + dc
			if not (0 <= nr2 < N and 0 <= nc2 < N) or grid[nr2][nc2] == '#':
				nr2, nc2 = r2, c2
				
			if (nr1, nc1) <= (nr2, nc2):
				new_state = (nr1, nc1, nr2, nc2)
			else:
				new_state = (nr2, nc2, nr1, nc1)
				
			if not visited[new_state[0]][new_state[1]][new_state[2]][new_state[3]]:
				visited[new_state[0]][new_state[1]][new_state[2]][new_state[3]] = True
				q.append((new_state[0], new_state[1], new_state[2], new_state[3], steps+1))
				
	if not found:
		print(-1)

if __name__ == "__main__":
	main()
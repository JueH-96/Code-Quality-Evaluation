import collections

def main():
	H, W = map(int, input().split())
	grid = [input().strip() for _ in range(H)]
	
	pattern = "snuke"
	if grid[0][0] != pattern[0]:
		print("No")
		return
		
	visited = [[[False] * 5 for _ in range(W)] for __ in range(H)]
	visited[0][0][0] = True
	q = collections.deque()
	q.append((0, 0, 0))
	
	while q:
		i, j, mod = q.popleft()
		if i == H - 1 and j == W - 1:
			print("Yes")
			return
			
		next_mod = (mod + 1) % 5
		for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
			ni, nj = i + dx, j + dy
			if 0 <= ni < H and 0 <= nj < W:
				if grid[ni][nj] == pattern[next_mod]:
					if not visited[ni][nj][next_mod]:
						visited[ni][nj][next_mod] = True
						q.append((ni, nj, next_mod))
						
	print("No")

if __name__ == "__main__":
	main()
import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(-1)
		return
		
	H, W, K = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	ans = 10**9
	
	if K <= W:
		for i in range(H):
			total_x = 0
			total_dot = 0
			for j in range(K):
				if grid[i][j] == 'x':
					total_x += 1
				elif grid[i][j] == '.':
					total_dot += 1
					
			if total_x == 0:
				ans = min(ans, total_dot)
				if ans == 0:
					break
					
			for j in range(1, W - K + 1):
				if grid[i][j-1] == 'x':
					total_x -= 1
				elif grid[i][j-1] == '.':
					total_dot -= 1
					
				if grid[i][j+K-1] == 'x':
					total_x += 1
				elif grid[i][j+K-1] == '.':
					total_dot += 1
					
				if total_x == 0:
					ans = min(ans, total_dot)
					if ans == 0:
						break
			if ans == 0:
				break
				
	if ans == 0:
		print(0)
		return
		
	if K <= H:
		for j in range(W):
			total_x = 0
			total_dot = 0
			for i in range(K):
				if grid[i][j] == 'x':
					total_x += 1
				elif grid[i][j] == '.':
					total_dot += 1
					
			if total_x == 0:
				ans = min(ans, total_dot)
				if ans == 0:
					break
					
			for i in range(1, H - K + 1):
				if grid[i-1][j] == 'x':
					total_x -= 1
				elif grid[i-1][j] == '.':
					total_dot -= 1
					
				if grid[i+K-1][j] == 'x':
					total_x += 1
				elif grid[i+K-1][j] == '.':
					total_dot += 1
					
				if total_x == 0:
					ans = min(ans, total_dot)
					if ans == 0:
						break
			if ans == 0:
				break
				
	if ans == 10**9:
		print(-1)
	else:
		print(ans)

if __name__ == "__main__":
	main()
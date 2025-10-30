def main():
	n = int(input().strip())
	grid = [[False] * 100 for _ in range(100)]
	
	for _ in range(n):
		A, B, C, D = map(int, input().split())
		for x in range(A, B):
			for y in range(C, D):
				grid[x][y] = True
				
	total = 0
	for i in range(100):
		for j in range(100):
			if grid[i][j]:
				total += 1
				
	print(total)

if __name__ == "__main__":
	main()
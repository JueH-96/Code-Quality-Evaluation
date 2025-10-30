def main():
	N = int(input().strip())
	grid_A = []
	for _ in range(N):
		grid_A.append(input().strip())
	
	grid_B = []
	for _ in range(N):
		grid_B.append(input().strip())
	
	for i in range(N):
		for j in range(N):
			if grid_A[i][j] != grid_B[i][j]:
				print(f"{i+1} {j+1}")
				return

if __name__ == '__main__':
	main()
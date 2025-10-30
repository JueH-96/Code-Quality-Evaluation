def main():
	N = int(input().strip())
	matrix = []
	for i in range(N):
		row = list(map(int, input().split()))
		matrix.append(row)
	
	current = 1
	for next_elem in range(1, N + 1):
		a = current
		b = next_elem
		if a >= b:
			current = matrix[a - 1][b - 1]
		else:
			current = matrix[b - 1][a - 1]
	
	print(current)

if __name__ == "__main__":
	main()
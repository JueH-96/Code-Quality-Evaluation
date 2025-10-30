def main():
	n = int(input().strip())
	mat = []
	for i in range(n):
		row = list(map(int, input().split()))
		mat.append(row)
	
	current = 1
	for k in range(1, n + 1):
		if current >= k:
			current = mat[current - 1][k - 1]
		else:
			current = mat[k - 1][current - 1]
	
	print(current)

if __name__ == "__main__":
	main()
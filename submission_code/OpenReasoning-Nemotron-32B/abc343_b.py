n = int(input().strip())
for i in range(n):
	row = list(map(int, input().split()))
	neighbors = []
	for j in range(n):
		if row[j] == 1:
			neighbors.append(str(j + 1))
	print(" ".join(neighbors))
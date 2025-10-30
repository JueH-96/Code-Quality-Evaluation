n = int(input().strip())
graph = []
for _ in range(n):
	row = list(map(int, input().split()))
	graph.append(row)

for i in range(n):
	neighbors = []
	for j in range(n):
		if graph[i][j] == 1:
			neighbors.append(str(j + 1))
	print(" ".join(neighbors))
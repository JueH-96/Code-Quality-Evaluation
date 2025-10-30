def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	graph = [[False] * n for _ in range(n)]
	
	index = 2
	for _ in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		graph[a-1][b-1] = True
		
	for k in range(n):
		for i in range(n):
			for j in range(n):
				if graph[i][k] and graph[k][j]:
					graph[i][j] = True
					
	candidates = []
	for i in range(n):
		valid = True
		for j in range(n):
			if i != j and not graph[i][j]:
				valid = False
				break
		if valid:
			candidates.append(i)
			
	if len(candidates) == 1:
		print(candidates[0] + 1)
	else:
		print(-1)

if __name__ == "__main__":
	main()
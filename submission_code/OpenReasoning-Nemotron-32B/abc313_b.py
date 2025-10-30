def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	closure = [[False] * (n+1) for _ in range(n+1)]
	
	index = 2
	for _ in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		closure[a][b] = True
	
	for k in range(1, n+1):
		for i in range(1, n+1):
			for j in range(1, n+1):
				if closure[i][k] and closure[k][j]:
					closure[i][j] = True
	
	candidates = []
	for i in range(1, n+1):
		has_superior = False
		for j in range(1, n+1):
			if i == j:
				continue
			if closure[j][i]:
				has_superior = True
				break
		if not has_superior:
			candidates.append(i)
	
	print(candidates[0] if len(candidates) == 1 else -1)

if __name__ == '__main__':
	main()
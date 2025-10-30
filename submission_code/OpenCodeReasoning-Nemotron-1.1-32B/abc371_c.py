import itertools

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	M_G = int(next(it))
	adjG = [[0] * N for _ in range(N)]
	for _ in range(M_G):
		u = int(next(it)) - 1
		v = int(next(it)) - 1
		adjG[u][v] = 1
		adjG[v][u] = 1

	M_H = int(next(it))
	adjH = [[0] * N for _ in range(N)]
	for _ in range(M_H):
		a = int(next(it)) - 1
		b = int(next(it)) - 1
		adjH[a][b] = 1
		adjH[b][a] = 1

	cost_mat = [[0] * N for _ in range(N)]
	for i in range(N - 1):
		for j in range(i + 1, N):
			val = int(next(it))
			cost_mat[i][j] = val

	permutations = list(itertools.permutations(range(N)))
	ans = float('inf')
	
	for p in permutations:
		target = [[0] * N for _ in range(N)]
		for i in range(N):
			for j in range(i + 1, N):
				if adjG[i][j]:
					a = p[i]
					b = p[j]
					target[a][b] = 1
					target[b][a] = 1
		
		cost_here = 0
		for i in range(N):
			for j in range(i + 1, N):
				if adjH[i][j] != target[i][j]:
					cost_here += cost_mat[i][j]
		
		if cost_here < ans:
			ans = cost_here
	
	print(ans)

if __name__ == '__main__':
	main()
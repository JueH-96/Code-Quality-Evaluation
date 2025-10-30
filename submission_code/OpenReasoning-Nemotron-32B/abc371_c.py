import itertools

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	M_G = int(next(it))
	G_set = set()
	for _ in range(M_G):
		u = int(next(it))
		v = int(next(it))
		if u > v:
			u, v = v, u
		G_set.add((u, v))
	
	M_H = int(next(it))
	H_set = set()
	for _ in range(M_H):
		u = int(next(it))
		v = int(next(it))
		if u > v:
			u, v = v, u
		H_set.add((u, v))
	
	A_mat = [[0] * (N+1) for _ in range(N+1)]
	for i in range(1, N):
		num_vals = N - i
		row = []
		for _ in range(num_vals):
			row.append(int(next(it)))
		for j in range(i+1, N+1):
			idx = j - i - 1
			A_mat[i][j] = row[idx]
			A_mat[j][i] = row[idx]
	
	perms = itertools.permutations(range(1, N+1))
	ans = float('inf')
	
	for p in perms:
		total_cost = 0
		for i in range(1, N+1):
			for j in range(i+1, N+1):
				u = p[i-1]
				v = p[j-1]
				a = min(u, v)
				b = max(u, v)
				edge_in_G = (i, j) in G_set
				edge_in_H = (a, b) in H_set
				if edge_in_G != edge_in_H:
					total_cost += A_mat[a][b]
		if total_cost < ans:
			ans = total_cost
	
	print(ans)

if __name__ == "__main__":
	main()
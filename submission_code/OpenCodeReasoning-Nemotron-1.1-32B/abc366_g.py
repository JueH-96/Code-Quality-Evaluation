import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	
	deg = [0] * n
	graph = [[0] * n for _ in range(n)]
	
	for _ in range(m):
		u = int(next(it))
		v = int(next(it))
		u0 = u - 1
		v0 = v - 1
		deg[u0] += 1
		deg[v0] += 1
		graph[u0][v0] = 1
		graph[v0][u0] = 1
		
	if any(d == 1 for d in deg):
		print("No")
		return
		
	M_mat = [row[:] for row in graph]
	pivot_row = [-1] * n
	pivot_col = [-1] * n
	r = 0
	
	for j in range(n):
		found = -1
		for i in range(r, n):
			if M_mat[i][j] == 1:
				found = i
				break
		if found == -1:
			continue
		M_mat[r], M_mat[found] = M_mat[found], M_mat[r]
		pivot_row[j] = r
		pivot_col[r] = j
		for i in range(r + 1, n):
			if M_mat[i][j] == 1:
				for k in range(n):
					M_mat[i][k] ^= M_mat[r][k]
		r += 1
		
	free_columns = []
	for j in range(n):
		if pivot_row[j] == -1:
			free_columns.append(j)
			
	if len(free_columns) == 0:
		print("No")
		return
		
	basis = []
	for j in free_columns:
		v = [0] * n
		v[j] = 1
		for i in range(r):
			v[pivot_col[i]] = M_mat[i][j]
		basis.append(v)
		
	res = [0] * n
	for idx, j in enumerate(free_columns):
		constant = 1 << j
		vec = basis[idx]
		for i in range(n):
			if vec[i] == 1:
				res[i] += constant
				
	print("Yes")
	print(" ".join(map(str, res)))
	
if __name__ == "__main__":
	main()
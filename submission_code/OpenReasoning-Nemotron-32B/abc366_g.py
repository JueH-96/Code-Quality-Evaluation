def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n)]
	deg = [0] * n
	index = 2
	for _ in range(m):
		u = int(data[index])
		v = int(data[index+1])
		index += 2
		u -= 1
		v -= 1
		graph[u].append(v)
		graph[v].append(u)
		deg[u] += 1
		deg[v] += 1
	
	if any(d == 1 for d in deg):
		print("No")
		return
		
	A = []
	for i in range(n):
		if deg[i] == 0:
			continue
		row = [0] * n
		for neighbor in graph[i]:
			row[neighbor] = 1
		A.append(row)
	
	rows = len(A)
	cols = n
	if rows == 0:
		basis_vectors = []
		for i in range(n):
			vec = [0] * n
			vec[i] = 1
			basis_vectors.append(vec)
		k = len(basis_vectors)
		res = [0] * n
		for j in range(k):
			for i in range(n):
				if basis_vectors[j][i] == 1:
					res[i] |= (1 << j)
		print("Yes")
		print(" ".join(map(str, res)))
		return
		
	M = [row[:] for row in A]
	pivot_cols = []
	pivot_row_for_col = [-1] * cols
	r = 0
	for col in range(cols):
		pivot = -1
		for i in range(r, rows):
			if M[i][col] == 1:
				pivot = i
				break
		if pivot == -1:
			continue
		M[r], M[pivot] = M[pivot], M[r]
		pivot_cols.append(col)
		pivot_row_for_col[col] = r
		for i in range(rows):
			if i != r and M[i][col] == 1:
				for j in range(cols):
					M[i][j] ^= M[r][j]
		r += 1
		
	free_cols = [col for col in range(cols) if col not in pivot_cols]
	
	for i in range(n):
		if i in free_cols:
			continue
		else:
			r_index = pivot_row_for_col[i]
			if r_index == -1:
				continue
			row = M[r_index]
			found = False
			for j in free_cols:
				if row[j] == 1:
					found = True
					break
			if not found:
				print("No")
				return
				
	basis_vectors = []
	for f in free_cols:
		vec = [0] * n
		vec[f] = 1
		for col in pivot_cols:
			r_index = pivot_row_for_col[col]
			vec[col] = M[r_index][f]
		basis_vectors.append(vec)
		
	k = len(basis_vectors)
	res = [0] * n
	for j in range(k):
		for i in range(n):
			if basis_vectors[j][i] == 1:
				res[i] |= (1 << j)
				
	print("Yes")
	print(" ".join(map(str, res)))
	
if __name__ == "__main__":
	main()
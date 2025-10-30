import itertools

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	H = int(data[0])
	W = int(data[1])
	A = []
	index = 2
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		A.append(row)
		index += W
		
	B = []
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		B.append(row)
		index += W
		
	flatA = sorted([x for row in A for x in row])
	flatB = sorted([x for row in B for x in row])
	if flatA != flatB:
		print(-1)
		return
		
	row_multiset_A = [sorted(row) for row in A]
	row_multiset_B = [sorted(row) for row in B]
	
	col_multiset_A = []
	for j in range(W):
		col = [A[i][j] for i in range(H)]
		col_multiset_A.append(sorted(col))
		
	col_multiset_B = []
	for j in range(W):
		col = [B[i][j] for i in range(H)]
		col_multiset_B.append(sorted(col))
		
	row_perms = list(itertools.permutations(range(H)))
	col_perms = list(itertools.permutations(range(W)))
	
	def inv_count(perm):
		cnt = 0
		n = len(perm)
		for i in range(n):
			for j in range(i+1, n):
				if perm[i] > perm[j]:
					cnt += 1
		return cnt
		
	min_ops = 10**9
	found = False
	for p in row_perms:
		skip_p = False
		for i in range(H):
			if row_multiset_A[p[i]] != row_multiset_B[i]:
				skip_p = True
				break
		if skip_p:
			continue
			
		for q in col_perms:
			skip_q = False
			for j in range(W):
				if col_multiset_A[q[j]] != col_multiset_B[j]:
					skip_q = True
					break
			if skip_q:
				continue
				
			valid = True
			for i in range(H):
				for j in range(W):
					if A[p[i]][q[j]] != B[i][j]:
						valid = False
						break
				if not valid:
					break
					
			if valid:
				found = True
				cost = inv_count(p) + inv_count(q)
				if cost < min_ops:
					min_ops = cost
					
	if found:
		print(min_ops)
	else:
		print(-1)

if __name__ == "__main__":
	main()
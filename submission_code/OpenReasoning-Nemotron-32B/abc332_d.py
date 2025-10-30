import itertools

def main():
	import sys
	data = sys.stdin.read().strip().split()
	if not data:
		return
	
	H = int(data[0])
	W = int(data[1])
	A = []
	index = 2
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		index += W
		A.append(row)
	
	B = []
	for i in range(H):
		row = list(map(int, data[index:index+W]))
		index += W
		B.append(row)
	
	flat_A = [elem for row in A for elem in row]
	flat_B = [elem for row in B for elem in row]
	if sorted(flat_A) != sorted(flat_B):
		print(-1)
		return
		
	perms_rows = list(itertools.permutations(range(H)))
	perms_cols = list(itertools.permutations(range(W)))
	
	min_ops = float('inf')
	
	def inversion_count(perm):
		n = len(perm)
		cnt = 0
		for i in range(n):
			for j in range(i+1, n):
				if perm[i] > perm[j]:
					cnt += 1
		return cnt
		
	for p in perms_rows:
		for q in perms_cols:
			valid = True
			for i in range(H):
				for j in range(W):
					if A[p[i]][q[j]] != B[i][j]:
						valid = False
						break
				if not valid:
					break
			if valid:
				row_swaps = inversion_count(p)
				col_swaps = inversion_count(q)
				total_swaps = row_swaps + col_swaps
				if total_swaps < min_ops:
					min_ops = total_swaps
					
	if min_ops == float('inf'):
		print(-1)
	else:
		print(min_ops)

if __name__ == '__main__':
	main()
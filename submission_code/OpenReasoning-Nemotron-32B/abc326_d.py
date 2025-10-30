import itertools

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	N = int(data[0].strip())
	R = data[1].strip()
	C = data[2].strip()
	
	counts = [0] * N
	sets_so_far = [None] * N

	def dfs(i, counts, sets_so_far):
		if i == N:
			if all(c == 3 for c in counts):
				topmost = [None] * N
				for j in range(N):
					for i0 in range(N):
						if j in sets_so_far[i0]:
							topmost[j] = i0
							break
				grid = [['.' for _ in range(N)] for _ in range(N)]
				col_letters = [set() for _ in range(N)]
				for row in range(N):
					s = sets_so_far[row]
					j0 = min(s)
					if row == topmost[j0]:
						if R[row] != C[j0]:
							return None
					if R[row] in col_letters[j0]:
						return None
					grid[row][j0] = R[row]
					col_letters[j0].add(R[row])
					
					other_cols = list(s - {j0})
					forced = []
					free = []
					for j in other_cols:
						if row == topmost[j]:
							forced.append(j)
						else:
							free.append(j)
					
					available_letters = set(['A','B','C']) - {R[row]}
					
					for j in forced:
						if C[j] not in available_letters:
							return None
						if C[j] in col_letters[j]:
							return None
						grid[row][j] = C[j]
						col_letters[j].add(C[j])
						available_letters.discard(C[j])
					
					if free:
						perms = list(itertools.permutations(available_letters, len(free)))
						found = False
						for p in perms:
							valid = True
							for idx, j in enumerate(free):
								if p[idx] in col_letters[j]:
									valid = False
									break
							if not valid:
								continue
							for idx, j in enumerate(free):
								grid[row][j] = p[idx]
								col_letters[j].add(p[idx])
							found = True
							break
						if not found:
							return None
				return grid
			else:
				return None
				
		for j0 in range(N):
			avail = [j for j in range(j0+1, N)]
			if len(avail) < 2:
				continue
			for c1, c2 in itertools.combinations(avail, 2):
				s = {j0, c1, c2}
				counts[j0] += 1
				counts[c1] += 1
				counts[c2] += 1
				if counts[j0] > 3 or counts[c1] > 3 or counts[c2] > 3:
					counts[j0] -= 1
					counts[c1] -= 1
					counts[c2] -= 1
					continue
				sets_so_far[i] = s
				res = dfs(i+1, counts, sets_so_far)
				if res is not None:
					return res
				counts[j0] -= 1
				counts[c1] -= 1
				counts[c2] -= 1
		return None

	result_grid = dfs(0, counts, sets_so_far)
	if result_grid is not None:
		print("Yes")
		for row in result_grid:
			print(''.join(row))
	else:
		print("No")

if __name__ == '__main__':
	main()
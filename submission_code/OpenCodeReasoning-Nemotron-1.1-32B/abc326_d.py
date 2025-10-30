import sys
import itertools

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	N = int(data[0].strip())
	R = data[1].strip()
	C = data[2].strip()
	
	grid = [['.'] * N for _ in range(N)]
	col_sets = [set() for _ in range(N)]
	col_count = [0] * N

	def dfs(i):
		if i == N:
			return grid
		
		rem_letters = list({'A','B','C'} - {R[i]})
		for cols in itertools.combinations(range(N), 3):
			for perm in itertools.permutations(rem_letters):
				j1, j2, j3 = cols
				letter1 = R[i]
				letter2 = perm[0]
				letter3 = perm[1]
				
				valid = True
				if col_count[j1] >= 3:
					valid = False
				elif col_count[j1] == 0:
					if letter1 != C[j1]:
						valid = False
				elif letter1 in col_sets[j1]:
					valid = False
					
				if not valid:
					continue
					
				if col_count[j2] >= 3:
					valid = False
				elif col_count[j2] == 0:
					if letter2 != C[j2]:
						valid = False
				elif letter2 in col_sets[j2]:
					valid = False
					
				if not valid:
					continue
					
				if col_count[j3] >= 3:
					valid = False
				elif col_count[j3] == 0:
					if letter3 != C[j3]:
						valid = False
				elif letter3 in col_sets[j3]:
					valid = False
					
				if not valid:
					continue
				
				grid[i][j1] = letter1
				grid[i][j2] = letter2
				grid[i][j3] = letter3
				
				col_sets[j1].add(letter1)
				col_sets[j2].add(letter2)
				col_sets[j3].add(letter3)
				col_count[j1] += 1
				col_count[j2] += 1
				col_count[j3] += 1
				
				res = dfs(i+1)
				if res is not None:
					return res
					
				grid[i][j1] = '.'
				grid[i][j2] = '.'
				grid[i][j3] = '.'
				
				col_sets[j1].remove(letter1)
				col_sets[j2].remove(letter2)
				col_sets[j3].remove(letter3)
				col_count[j1] -= 1
				col_count[j2] -= 1
				col_count[j3] -= 1
				
		return None
		
	result = dfs(0)
	if result is None:
		print("No")
	else:
		print("Yes")
		for row in result:
			print(''.join(row))

if __name__ == "__main__":
	main()
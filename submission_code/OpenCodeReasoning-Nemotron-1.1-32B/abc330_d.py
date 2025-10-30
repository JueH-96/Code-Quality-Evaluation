import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	n = int(data[0])
	grid = data[1:1+n]
	
	row_ones = []
	for i in range(n):
		row = []
		for j in range(n):
			if grid[i][j] == 'o':
				row.append(j)
		row_ones.append(row)
	
	col_pair_count = [[0] * n for _ in range(n)]
	
	for i in range(n):
		cols = row_ones[i]
		k = len(cols)
		for idx1 in range(k):
			for idx2 in range(idx1+1, k):
				j1 = cols[idx1]
				j2 = cols[idx2]
				if j1 > j2:
					j1, j2 = j2, j1
				col_pair_count[j1][j2] += 1
				
	total = 0
	for i in range(n):
		for j in range(n):
			if grid[i][j] == 'x':
				for j2 in row_ones[i]:
					if j2 == j:
						continue
					a, b = j, j2
					if a > b:
						a, b = b, a
					total += col_pair_count[a][b]
					
	print(total)

if __name__ == "__main__":
	main()
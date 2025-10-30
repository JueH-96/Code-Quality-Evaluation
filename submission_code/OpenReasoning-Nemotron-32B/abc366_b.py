def main():
	n = int(input().strip())
	strings = [input().strip() for _ in range(n)]
	
	if n == 0:
		return
	
	M = max(len(s) for s in strings)
	grid = [['*'] * M for _ in range(n)]
	
	for i in range(n):
		s = strings[i]
		row_index = n - 1 - i
		for j in range(len(s)):
			grid[row_index][j] = s[j]
	
	for j in range(M):
		col_str = ''.join(grid[i][j] for i in range(n))
		col_str = col_str.rstrip('*')
		print(col_str)

if __name__ == '__main__':
	main()
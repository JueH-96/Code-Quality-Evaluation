def main():
	n = int(input().strip())
	grid = [input().strip() for _ in range(n)]
	res = [[''] * n for _ in range(n)]
	
	for i in range(n):
		for j in range(n):
			d = min(i, j, n-1-i, n-1-j)
			k = (d + 1) % 4
			if k == 0:
				a, b = i, j
			elif k == 1:
				a, b = n-1 - j, i
			elif k == 2:
				a, b = n-1 - i, n-1 - j
			else:
				a, b = j, n-1 - i
			res[i][j] = grid[a][b]
	
	for row in res:
		print(''.join(row))

if __name__ == '__main__':
	main()
def main():
	import sys
	data = sys.stdin.read().split()
	H = int(data[0])
	W = int(data[1])
	N = int(data[2])
	
	grid = [['.' for _ in range(W)] for _ in range(H)]
	r, c = 0, 0
	d = 0
	dr = [-1, 0, 1, 0]
	dc = [0, 1, 0, -1]
	
	for _ in range(N):
		if grid[r][c] == '.':
			grid[r][c] = '#'
			d = (d + 1) % 4
		else:
			grid[r][c] = '.'
			d = (d + 3) % 4
		
		r = (r + dr[d]) % H
		c = (c + dc[d]) % W
	
	for row in grid:
		print(''.join(row))

if __name__ == '__main__':
	main()
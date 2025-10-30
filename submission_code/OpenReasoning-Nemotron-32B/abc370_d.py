import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	it = iter(data)
	H = int(next(it)); W = int(next(it)); Q = int(next(it))
	
	active = [[True] * W for _ in range(H)]
	next_above = [[0] * W for _ in range(H)]
	next_below = [[0] * W for _ in range(H)]
	next_left = [[0] * W for _ in range(H)]
	next_right = [[0] * W for _ in range(H)]
	
	for i in range(H):
		for j in range(W):
			next_above[i][j] = i - 1
			next_below[i][j] = i + 1
			next_left[i][j] = j - 1
			next_right[i][j] = j + 1
			
	for i in range(H):
		for j in range(W):
			if i == 0:
				next_above[i][j] = -1
			if i == H - 1:
				next_below[i][j] = -1
			if j == 0:
				next_left[i][j] = -1
			if j == W - 1:
				next_right[i][j] = -1
				
	def find_above(i, j):
		if i < 0:
			return -1
		path = []
		cur = i
		while cur >= 0 and not active[cur][j]:
			path.append((cur, j))
			if next_above[cur][j] == cur - 1:
				cur = cur - 1
			else:
				cur = next_above[cur][j]
		if cur < 0:
			result = -1
		else:
			result = cur
		for (x, y) in path:
			next_above[x][y] = result
		return result

	def find_below(i, j):
		if i >= H:
			return -1
		path = []
		cur = i
		while cur < H and not active[cur][j]:
			path.append((cur, j))
			if next_below[cur][j] == cur + 1:
				cur = cur + 1
			else:
				cur = next_below[cur][j]
		if cur >= H:
			result = -1
		else:
			result = cur
		for (x, y) in path:
			next_below[x][y] = result
		return result

	def find_left(i, j):
		if j < 0:
			return -1
		path = []
		cur = j
		while cur >= 0 and not active[i][cur]:
			path.append((i, cur))
			if next_left[i][cur] == cur - 1:
				cur = cur - 1
			else:
				cur = next_left[i][cur]
		if cur < 0:
			result = -1
		else:
			result = cur
		for (x, y) in path:
			next_left[x][y] = result
		return result

	def find_right(i, j):
		if j >= W:
			return -1
		path = []
		cur = j
		while cur < W and not active[i][cur]:
			path.append((i, cur))
			if next_right[i][cur] == cur + 1:
				cur = cur + 1
			else:
				cur = next_right[i][cur]
		if cur >= W:
			result = -1
		else:
			result = cur
		for (x, y) in path:
			next_right[x][y] = result
		return result

	for _ in range(Q):
		r = int(next(it)); c = int(next(it))
		r -= 1; c -= 1
		if active[r][c]:
			active[r][c] = False
		else:
			walls = []
			up_wall = find_above(r - 1, c)
			if up_wall != -1:
				walls.append((up_wall, c))
				
			down_wall = find_below(r + 1, c)
			if down_wall != -1:
				walls.append((down_wall, c))
				
			left_wall = find_left(r, c - 1)
			if left_wall != -1:
				walls.append((r, left_wall))
				
			right_wall = find_right(r, c + 1)
			if right_wall != -1:
				walls.append((r, right_wall))
				
			for (i, j) in walls:
				if active[i][j]:
					active[i][j] = False
					
	count = 0
	for i in range(H):
		for j in range(W):
			if active[i][j]:
				count += 1
				
	print(count)

if __name__ == '__main__':
	main()
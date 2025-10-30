class Solution:
	def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
		M = 12345
		n = len(grid)
		m = len(grid[0])
		L = n * m
		arr = []
		for i in range(n):
			for j in range(m):
				arr.append(grid[i][j])
		
		size = 1
		while size < L:
			size *= 2
		seg = [1] * (2 * size)
		
		for i in range(L):
			seg[size + i] = arr[i] % M
		
		for i in range(size - 1, 0, -1):
			seg[i] = (seg[2*i] * seg[2*i+1]) % M
		
		def query(l, r):
			res = 1
			l += size
			r += size
			while l <= r:
				if l % 2 == 1:
					res = (res * seg[l]) % M
					l += 1
				if r % 2 == 0:
					res = (res * seg[r]) % M
					r -= 1
				l //= 2
				r //= 2
			return res
		
		res_arr = [0] * L
		for i in range(L):
			left_part = 1
			if i > 0:
				left_part = query(0, i-1)
			right_part = 1
			if i < L - 1:
				right_part = query(i+1, L-1)
			res_arr[i] = (left_part * right_part) % M
		
		res = []
		for i in range(n):
			res.append(res_arr[i*m: (i+1)*m])
		return res
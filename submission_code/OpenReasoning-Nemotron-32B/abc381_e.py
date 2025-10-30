import sys

class SegmentTree:
	def __init__(self, n, mode):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.mode = mode
		if mode == 'max':
			self.default = -10**18
		else:
			self.default = 10**18
		self.data = [self.default] * (2 * self.size)
	
	def update(self, index, value):
		i = index + self.size
		if self.mode == 'max':
			if value > self.data[i]:
				self.data[i] = value
		else:
			if value < self.data[i]:
				self.data[i] = value
		i //= 2
		while i:
			if self.mode == 'max':
				self.data[i] = max(self.data[2*i], self.data[2*i+1])
			else:
				self.data[i] = min(self.data[2*i], self.data[2*i+1])
			i //= 2
			
	def query(self, l, r):
		l += self.size
		r += self.size
		if self.mode == 'max':
			res = self.default
			while l <= r:
				if l % 2 == 1:
					res = max(res, self.data[l])
					l += 1
				if r % 2 == 0:
					res = max(res, self.data[r])
					r -= 1
				l //= 2
				r //= 2
			return res
		else:
			res = self.default
			while l <= r:
				if l % 2 == 1:
					res = min(res, self.data[l])
					l += 1
				if r % 2 == 0:
					res = min(res, self.data[r])
					r -= 1
				l //= 2
				r //= 2
			return res

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	q = int(data[1])
	S = data[2].strip()
	queries_input = data[3:3+2*q]
	
	prefix1 = [0] * (n+1)
	prefix2 = [0] * (n+1)
	for i in range(1, n+1):
		c = S[i-1]
		prefix1[i] = prefix1[i-1] + (1 if c == '1' else 0)
		prefix2[i] = prefix2[i-1] + (1 if c == '2' else 0)
	
	D = []
	for i in range(n):
		if S[i] == '/':
			D.append(i)
	
	points = []
	for i in D:
		x_i = prefix1[i]
		y_i = prefix2[i+1]
		key_i = x_i + y_i
		points.append((key_i, i, x_i, y_i))
	
	queries = []
	for i in range(q):
		L = int(queries_input[2*i])
		R = int(queries_input[2*i+1])
		L0 = L - 1
		R0 = R - 1
		c1 = prefix1[L0]
		d2 = prefix2[R0+1]
		threshold = c1 + d2
		queries.append((L0, R0, c1, d2, threshold))
	
	if not D:
		for _ in range(q):
			print(0)
		return
	
	events1 = []
	for (key_i, i, x_i, y_i) in points:
		events1.append((key_i, i, x_i, y_i, 'point'))
	for j, (L0, R0, c1, d2, threshold) in enumerate(queries):
		events1.append((threshold, L0, R0, j, 'query'))
	
	events1.sort(key=lambda e: (e[0], 0 if e[-1]=='point' else 1))
	
	st_max = SegmentTree(n, 'max')
	ans1 = [-10**18] * q
	
	for event in events1:
		if event[-1] == 'point':
			key_i, i, x_i, y_i, _ = event
			st_max.update(i, x_i)
		else:
			threshold_val, L0, R0, j, _ = event
			max_x = st_max.query(L0, R0)
			if max_x > -10**18:
				c1_val = queries[j][2]
				ans1[j] = 2 * max_x - 2 * c1_val + 1
	
	events2 = []
	for (key_i, i, x_i, y_i) in points:
		events2.append((key_i, i, x_i, y_i, 'point'))
	for j, (L0, R0, c1, d2, threshold) in enumerate(queries):
		events2.append((threshold, L0, R0, j, 'query'))
	
	events2.sort(key=lambda e: (-e[0], 0 if e[-1]=='query' else 1))
	
	st_min = SegmentTree(n, 'min')
	ans2 = [-10**18] * q
	
	for event in events2:
		if event[-1] == 'point':
			key_i, i, x_i, y_i, _ = event
			st_min.update(i, y_i)
		else:
			threshold_val, L0, R0, j, _ = event
			min_y = st_min.query(L0, R0)
			if min_y < 10**18:
				d2_val = queries[j][3]
				ans2[j] = 2 * d2_val - 2 * min_y + 1
	
	for j in range(q):
		res = max(0, ans1[j], ans2[j])
		print(res)

if __name__ == "__main__":
	main()
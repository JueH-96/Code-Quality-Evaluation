import sys
import bisect

class Fenwick1D:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, delta):
		i = index + 1
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def query(self, index):
		res = 0
		i = index + 1
		while i:
			res += self.tree[i]
			i -= i & -i
		return res

	def query_range(self, l, r):
		if l > r:
			return 0
		res = self.query(r)
		if l > 0:
			res -= self.query(l - 1)
		return res

class Fenwick2D:
	def __init__(self, size_value, size_index):
		self.size_value = size_value
		self.size_index = size_index
		self.trees = [None] * (size_value + 1)
		for i in range(1, size_value + 1):
			self.trees[i] = Fenwick1D(size_index)
	
	def update(self, v, i, delta):
		v0 = v + 1
		while v0 <= self.size_value:
			self.trees[v0].update(i, delta)
			v0 += v0 & -v0
			
	def query_value_index_range(self, v_low, v_high, i_low, i_high):
		if v_low > v_high or i_low > i_high:
			return 0
		return self._query(v_high, i_low, i_high) - self._query(v_low - 1, i_low, i_high)
	
	def _query(self, v, i_low, i_high):
		res = 0
		v0 = v + 1
		while v0:
			res += self.trees[v0].query_range(i_low, i_high)
			v0 -= v0 & -v0
		return res

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	a = list(map(int, data[2:2 + n]))
	
	fenw_t = Fenwick1D(n)
	T = 0
	for i in range(n - 1, -1, -1):
		T += fenw_t.query(a[i] - 1)
		fenw_t.update(a[i], 1)
	
	by_value = [[] for _ in range(m)]
	for i in range(n):
		if a[i] < m:
			by_value[a[i]].append(i)
	
	size_value = m
	size_index = n
	fenw2d_p1 = Fenwick2D(size_value, size_index)
	fenw2d_p2 = Fenwick2D(size_value, size_index)
	
	for i in range(n):
		fenw2d_p1.update(a[i], i, 1)
	
	res = [0] * m
	res[0] = T
	
	size_p1 = n
	size_p2 = 0
	inv_p1 = T
	inv_p2 = 0
	
	for X in range(m - 1, 0, -1):
		for i in by_value[X]:
			fenw2d_p1.update(X, i, -1)
			left_count = fenw2d_p1.query_value_index_range(X + 1, m - 1, 0, i - 1)
			right_count = fenw2d_p1.query_value_index_range(0, X - 1, i + 1, n - 1)
			inv_p1 -= (left_count + right_count)
			
			fenw2d_p2.update(X, i, 1)
			left_count2 = fenw2d_p2.query_value_index_range(X + 1, m - 1, 0, i - 1)
			right_count2 = fenw2d_p2.query_value_index_range(0, X - 1, i + 1, n - 1)
			inv_p2 += (left_count2 + right_count2)
			
			size_p1 -= 1
			size_p2 += 1
		
		k = m - X
		if k < m:
			res[k] = 2 * (inv_p1 + inv_p2) + size_p1 * size_p2 - T
	
	for i in range(m):
		print(res[i])

if __name__ == '__main__':
	main()
import sys

class FenwTree:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, val):
		i = index + 1
		while i <= self.n:
			self.tree[i] ^= val
			i += i & -i
			
	def query(self, index):
		i = index + 1
		res = 0
		while i:
			res ^= self.tree[i]
			i -= i & -i
		return res

class SegmentTree:
	def __init__(self, data):
		self.n = len(data)
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.tree = [0] * (2 * self.size)
		for i in range(self.n):
			self.tree[self.size + i] = data[i]
		for i in range(self.size - 1, 0, -1):
			self.tree[i] = self.tree[2 * i] | self.tree[2 * i + 1]
			
	def update(self, index, value):
		i = self.size + index
		self.tree[i] = value
		i //= 2
		while i:
			self.tree[i] = self.tree[2 * i] | self.tree[2 * i + 1]
			i //= 2
			
	def query(self, l, r):
		l += self.size
		r += self.size
		res = 0
		while l <= r:
			if l % 2 == 1:
				res |= self.tree[l]
				l += 1
			if r % 2 == 0:
				res |= self.tree[r]
				r -= 1
			l //= 2
			r //= 2
		return res

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, q = map(int, data[0].split())
	s = data[1].strip()
	base = [1 if c == '1' else 0 for c in s]
	
	fenw = FenwTree(n)
	
	if n >= 2:
		B0 = []
		for i in range(n - 1):
			if base[i] == base[i + 1]:
				B0.append(1)
			else:
				B0.append(0)
		seg_tree = SegmentTree(B0)
	else:
		seg_tree = None
		
	out_lines = []
	index = 2
	for _ in range(q):
		parts = data[index].split()
		index += 1
		t = parts[0]
		L = int(parts[1])
		R = int(parts[2])
		if t == '1':
			l0 = L - 1
			r0 = R - 1
			fenw.update(l0, 1)
			if r0 + 1 < n:
				fenw.update(r0 + 1, 1)
				
			boundaries = set()
			if l0 - 1 >= 0 and l0 - 1 < n - 1:
				boundaries.add(l0 - 1)
			if r0 >= 0 and r0 < n - 1:
				boundaries.add(r0)
				
			for idx in boundaries:
				c1 = base[idx] ^ fenw.query(idx)
				c2 = base[idx + 1] ^ fenw.query(idx + 1)
				new_val = 1 if c1 == c2 else 0
				if n >= 2:
					seg_tree.update(idx, new_val)
		else:
			if R - L + 1 == 1:
				out_lines.append("Yes")
			else:
				l_idx = L - 1
				r_idx = R - 2
				if n >= 2:
					res = seg_tree.query(l_idx, r_idx)
					if res == 0:
						out_lines.append("Yes")
					else:
						out_lines.append("No")
				else:
					out_lines.append("Yes")
					
	for line in out_lines:
		print(line)

if __name__ == '__main__':
	main()
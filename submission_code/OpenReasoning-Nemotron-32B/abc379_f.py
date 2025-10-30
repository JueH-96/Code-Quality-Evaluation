import sys

class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (size + 1)
	
	def update(self, index, delta):
		i = index
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def query(self, index):
		s = 0
		i = index
		while i > 0:
			s += self.tree[i]
			i -= i & -i
		return s

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	H = [0]
	for _ in range(n):
		H.append(int(next(it)))
	
	L_arr = [0] * (n + 1)
	stack = []
	for i in range(1, n + 1):
		while stack and H[stack[-1]] < H[i]:
			stack.pop()
		if stack:
			L_arr[i] = stack[-1]
		else:
			L_arr[i] = 0
		stack.append(i)
	
	points = []
	for i in range(1, n + 1):
		points.append((L_arr[i], i))
	points.sort(key=lambda x: x[0])
	
	queries = []
	for i in range(q):
		l = int(next(it))
		r = int(next(it))
		queries.append((l, r, i))
	queries.sort(key=lambda x: x[0])
	
	fenw = Fenw(n)
	ptr = 0
	ans_arr = [0] * q
	
	for l, r, idx in queries:
		while ptr < len(points) and points[ptr][0] < l:
			L_val, k_val = points[ptr]
			fenw.update(k_val, 1)
			ptr += 1
		total = fenw.query(n) - fenw.query(r)
		ans_arr[idx] = total
	
	for ans in ans_arr:
		print(ans)

if __name__ == "__main__":
	main()
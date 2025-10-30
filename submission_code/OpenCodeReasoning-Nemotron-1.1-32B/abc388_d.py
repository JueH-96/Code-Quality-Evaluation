import sys

class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, delta):
		i = index + 1
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def prefix(self, index):
		if index < 0:
			return 0
		i = index + 1
		s = 0
		while i:
			s += self.tree[i]
			i -= i & -i
		return s
			
	def query(self, l, r):
		if l > r:
			return 0
		return self.prefix(r) - self.prefix(l - 1)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1 + n]))
	
	max_val = 1500000
	fenw = Fenw(max_val + 1)
	
	F = [0] * n
	for i in range(n):
		count = fenw.query(i, max_val)
		F[i] = A[i] + count
		key = F[i] + i
		if key <= max_val:
			fenw.update(key, 1)
			
	res = []
	for i in range(n):
		give = min(F[i], n - 1 - i)
		res.append(str(F[i] - give))
		
	print(" ".join(res))

if __name__ == "__main__":
	main()
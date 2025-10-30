import sys

class Fenw:
	def __init__(self, n):
		self.n = n
		self.tree = [0] * (n + 1)
	
	def update(self, index, delta):
		pos = index + 1
		while pos <= self.n:
			self.tree[pos] += delta
			pos += pos & -pos
			
	def query(self, index):
		if index < 0:
			return 0
		if index >= self.n:
			index = self.n - 1
		pos = index + 1
		s = 0
		while pos:
			s += self.tree[pos]
			pos -= pos & -pos
		return s

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1 + n]))
	
	sorted_A = sorted(set(A))
	comp = {val: idx for idx, val in enumerate(sorted_A)}
	size = len(sorted_A)
	
	fenw_left = Fenw(size)
	term1 = 0
	for i in range(n):
		num = A[i]
		idx = comp[num]
		count_less = fenw_left.query(idx - 1)
		term1 += num * count_less
		fenw_left.update(idx, 1)
		
	fenw_right = Fenw(size)
	term2 = 0
	for i in range(n - 1, -1, -1):
		num = A[i]
		idx = comp[num]
		count_less_equal = fenw_right.query(idx)
		count_greater = (n - 1 - i) - count_less_equal
		term2 += num * count_greater
		fenw_right.update(idx, 1)
		
	ans = term1 - term2
	print(ans)

if __name__ == '__main__':
	main()
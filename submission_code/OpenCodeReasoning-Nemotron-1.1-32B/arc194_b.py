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
			
	def query(self, index):
		res = 0
		i = index + 1
		while i > 0:
			res += self.tree[i]
			i -= i & -i
		return res

	def range_query(self, l, r):
		if l > r:
			return 0
		return self.query(r) - self.query(l - 1)

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	P = list(map(int, data[1:1 + n]))
	
	pos = [0] * (n + 1)
	for idx, num in enumerate(P):
		pos[num] = idx
	
	fenw = Fenw(n)
	total_cost = 0
	for i in range(1, n + 1):
		p = pos[i]
		count = fenw.range_query(p + 1, n - 1)
		current_pos = p + count
		move_steps = current_pos - (i - 1)
		if move_steps > 0:
			cost_i = (i + current_pos) * move_steps // 2
			total_cost += cost_i
		fenw.update(p, 1)
	
	print(total_cost)

if __name__ == "__main__":
	main()
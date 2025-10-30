import sys

class Fenw:
	def __init__(self, n):
		self.n = n
		self.fenw = [0] * (n + 1)
	
	def update(self, i, delta):
		idx = i + 1
		while idx <= self.n:
			self.fenw[idx] += delta
			idx += idx & -idx
			
	def prefix_sum(self, i):
		s = 0
		idx = i + 1
		while idx > 0:
			s += self.fenw[idx]
			idx -= idx & -idx
		return s
		
	def range_add(self, l, r, delta):
		self.update(l, delta)
		if r + 1 < self.n:
			self.update(r + 1, -delta)

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2 + n]))
	B = list(map(int, data[2 + n:2 + n + m]))
	
	fenw = Fenw(n)
	for i in range(n):
		fenw.range_add(i, i, A[i])
		
	global_cycles = 0
	for b in B:
		current_base = fenw.prefix_sum(b)
		k = current_base + global_cycles
		full_cycles = k // n
		r = k % n
		new_global_cycles = global_cycles + full_cycles
		
		fenw.range_add(b, b, -global_cycles - current_base)
		global_cycles = new_global_cycles
		
		if r > 0:
			start = (b + 1) % n
			last = (b + 1 + r - 1) % n
			if start <= last:
				fenw.range_add(start, last, 1)
			else:
				fenw.range_add(start, n - 1, 1)
				fenw.range_add(0, last, 1)
				
	result = []
	for i in range(n):
		base_val = fenw.prefix_sum(i)
		result.append(str(base_val + global_cycles))
		
	print(" ".join(result))

if __name__ == "__main__":
	main()
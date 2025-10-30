import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	P = [int(next(it)) for _ in range(n)]
	m = int(next(it))
	A = [int(next(it)) for _ in range(m)]
	
	size = n
	fenw = [0] * (size + 1)
	
	def update(i, delta):
		while i <= size:
			fenw[i] += delta
			i += i & -i
			
	def query(i):
		s = 0
		while i > 0:
			s += fenw[i]
			i -= i & -i
		return s
		
	inv = 0
	for i in range(n - 1, -1, -1):
		val = P[i]
		if val > 1:
			count = query(val - 1)
		else:
			count = 0
		inv += count
		update(val, 1)
	
	out_lines = []
	for k in A:
		swaps = 0
		for i in range(k - 1):
			if P[i] > P[i + 1]:
				P[i], P[i + 1] = P[i + 1], P[i]
				swaps += 1
		inv -= swaps
		out_lines.append(str(inv))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()
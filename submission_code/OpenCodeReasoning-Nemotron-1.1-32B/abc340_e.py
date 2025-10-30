import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	size = n + 1
	fenw = [0] * (size + 1)
	
	def update(i, delta):
		i1 = i + 1
		while i1 <= size:
			fenw[i1] += delta
			i1 += i1 & -i1

	def query(i):
		s = 0
		i1 = i + 1
		while i1:
			s += fenw[i1]
			i1 -= i1 & -i1
		return s

	global_add = 0
	ans_arr = [0] * n

	for i in range(m):
		b = B[i]
		v = A[b] + global_add + query(b)
		cycles = v // n
		rem = v % n
		global_add += cycles
		ans_arr[b] += cycles
		if rem > 0:
			l = (b + 1) % n
			r = (b + rem) % n
			if l <= r:
				update(l, 1)
				if r + 1 < n:
					update(r + 1, -1)
			else:
				update(l, 1)
				update(0, 1)
				if r + 1 < n:
					update(r + 1, -1)
					
	res = []
	for i in range(n):
		final_val = A[i] + global_add + query(i) - ans_arr[i]
		res.append(str(final_val))
		
	print(" ".join(res))

if __name__ == '__main__':
	main()
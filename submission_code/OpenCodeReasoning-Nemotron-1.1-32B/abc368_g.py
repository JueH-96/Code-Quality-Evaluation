import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	A = [0] * n
	for i in range(n):
		A[i] = int(next(it))
	B = [0] * n
	for i in range(n):
		B[i] = int(next(it))
	q = int(next(it))
	
	out_lines = []
	for _ in range(q):
		t = next(it)
		if t == '1':
			i_index = int(next(it)) - 1
			x = int(next(it))
			A[i_index] = x
		elif t == '2':
			i_index = int(next(it)) - 1
			x = int(next(it))
			B[i_index] = x
		else:
			l = int(next(it)) - 1
			r = int(next(it)) - 1
			v = 0
			for j in range(l, r + 1):
				if B[j] == 1:
					v += A[j]
				else:
					if v * (B[j] - 1) >= A[j]:
						v = v * B[j]
					else:
						v += A[j]
			out_lines.append(str(v))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()
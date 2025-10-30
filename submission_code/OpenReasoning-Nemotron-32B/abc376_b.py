def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, q = map(int, data[0].split())
	l = 1
	r = 2
	total_ops = 0
	index = 1
	for _ in range(q):
		parts = data[index].split()
		index += 1
		H = parts[0]
		T = int(parts[1])
		if H == 'L':
			d = (T - l) % n
			if d == 0:
				cost = 0
			else:
				x = (r - l) % n
				if x == 0:
					cost = d
				else:
					if 0 < x < d:
						cost = n - d
					else:
						cost = d
			total_ops += cost
			l = T
		else:
			d = (T - r) % n
			if d == 0:
				cost = 0
			else:
				x = (l - r) % n
				if x == 0:
					cost = d
				else:
					if 0 < x < d:
						cost = n - d
					else:
						cost = d
			total_ops += cost
			r = T
	print(total_ops)

if __name__ == '__main__':
	main()
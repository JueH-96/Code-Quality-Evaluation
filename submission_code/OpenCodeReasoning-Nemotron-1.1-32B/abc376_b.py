def main():
	import sys
	data = sys.stdin.read().splitlines()
	n, q = map(int, data[0].split())
	left = 1
	right = 2
	total_cost = 0
	index = 1
	for _ in range(q):
		parts = data[index].split()
		index += 1
		h = parts[0]
		t = int(parts[1])
		if h == 'L':
			A = left
			B = t
			O = right
		else:
			A = right
			B = t
			O = left
			
		if A == B:
			cost_i = 0
		else:
			d_cw = (B - A) % n
			d_O = (O - A) % n
			if d_O > 0 and d_O < d_cw:
				cost_i = n - d_cw
			else:
				cost_i = d_cw
				
		total_cost += cost_i
		
		if h == 'L':
			left = t
		else:
			right = t
			
	print(total_cost)

if __name__ == '__main__':
	main()
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	X = list(map(int, data[2:2+m]))
	
	base = 0
	diff = [0] * (n + 2)
	
	for i in range(m - 1):
		a = X[i]
		b = X[i + 1]
		d_clock = (b - a) % n
		if d_clock < 0:
			d_clock += n
		d_counter = n - d_clock
		min_val = min(d_clock, d_counter)
		base += min_val
		
		if 2 * min_val < n:
			v_i = n - 2 * min_val
			if d_clock == min_val:
				if a <= b:
					l1 = a
					r1 = b - 1
					if l1 <= r1:
						diff[l1] += v_i
						diff[r1 + 1] -= v_i
				else:
					l1 = a
					r1 = n
					if l1 <= r1:
						diff[l1] += v_i
						diff[r1 + 1] -= v_i
					l2 = 1
					r2 = b - 1
					if l2 <= r2:
						diff[l2] += v_i
						diff[r2 + 1] -= v_i
			else:
				if b <= a:
					l1 = b
					r1 = a - 1
					if l1 <= r1:
						diff[l1] += v_i
						diff[r1 + 1] -= v_i
				else:
					l1 = b
					r1 = n
					if l1 <= r1:
						diff[l1] += v_i
						diff[r1 + 1] -= v_i
					l2 = 1
					r2 = a - 1
					if l2 <= r2:
						diff[l2] += v_i
						diff[r2 + 1] -= v_i
	
	current = 0
	min_added = float('inf')
	for k in range(1, n + 1):
		current += diff[k]
		if current < min_added:
			min_added = current
	
	ans = base + min_added
	print(ans)

if __name__ == "__main__":
	main()
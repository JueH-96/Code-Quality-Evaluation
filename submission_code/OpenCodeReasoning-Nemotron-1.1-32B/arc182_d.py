import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	a = list(map(int, data[2:2+n]))
	b = list(map(int, data[2+n:2+2*n]))
	
	total_ops = 0
	for i in range(n):
		if a[i] == b[i]:
			continue
		d = (b[i] - a[i]) % m
		if d == 0:
			cost = 0
		else:
			if d <= m - d:
				next_val = (a[i] + 1) % m
				left_blocked = (i - 1 >= 0 and next_val == b[i-1])
				right_blocked = (i + 1 < n and next_val == a[i+1])
				if left_blocked or right_blocked:
					cost = m - d
				else:
					cost = d
			else:
				next_val = (a[i] - 1) % m
				if next_val < 0:
					next_val += m
				left_blocked = (i - 1 >= 0 and next_val == b[i-1])
				right_blocked = (i + 1 < n and next_val == a[i+1])
				if left_blocked or right_blocked:
					cost = d
				else:
					cost = m - d
		total_ops += cost
		
	print(total_ops)

if __name__ == '__main__':
	main()
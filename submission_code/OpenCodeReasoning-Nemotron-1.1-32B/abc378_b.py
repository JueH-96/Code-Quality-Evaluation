def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	garbage = []
	index = 1
	for _ in range(n):
		q = int(data[index])
		r = int(data[index + 1])
		index += 2
		garbage.append((q, r))
	
	q_count = int(data[index])
	index += 1
	queries = []
	for _ in range(q_count):
		t = int(data[index])
		d = int(data[index + 1])
		index += 2
		queries.append((t, d))
	
	for t, d in queries:
		q_val, r_val = garbage[t - 1]
		if d <= r_val:
			print(r_val)
		else:
			a = d - r_val
			k = (a - 1) // q_val + 1
			result = r_val + k * q_val
			print(result)

if __name__ == '__main__':
	main()
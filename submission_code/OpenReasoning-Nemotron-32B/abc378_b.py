def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	types = []
	index = 1
	for _ in range(n):
		q, r = map(int, data[index].split())
		types.append((q, r))
		index += 1
	
	q_count = int(data[index].strip())
	index += 1
	queries = []
	for _ in range(q_count):
		t, d = map(int, data[index].split())
		queries.append((t, d))
		index += 1
	
	results = []
	for t, d in queries:
		q_val, r_val = types[t-1]
		k0 = d // q_val
		candidate = k0 * q_val + r_val
		if candidate < d:
			candidate += q_val
		results.append(str(candidate))
	
	print("
".join(results))

if __name__ == '__main__':
	main()
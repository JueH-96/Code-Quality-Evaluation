import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	l_val = int(next(it))
	a = [int(next(it)) for _ in range(n)]
	b = [int(next(it)) for _ in range(m)]
	exclusions = []
	for _ in range(l_val):
		c = int(next(it))
		d = int(next(it))
		exclusions.append((c-1, d-1))
	
	excluded_main = [set() for _ in range(n)]
	for c, d in exclusions:
		excluded_main[c].add(d)
	
	a_indexed = [(a[i], i) for i in range(n)]
	a_indexed.sort(key=lambda x: x[0], reverse=True)
	k1 = min(l_val + 1, n)
	top_main_indices = set()
	for idx in range(k1):
		top_main_indices.add(a_indexed[idx][1])
	
	b_indexed = [(b[j], j) for j in range(m)]
	b_indexed.sort(key=lambda x: x[0], reverse=True)
	k2 = min(l_val + 1, m)
	top_side_indices = set()
	for idx in range(k2):
		top_side_indices.add(b_indexed[idx][1])
	
	excluded_side = [set() for _ in range(m)]
	for c, d in exclusions:
		excluded_side[d].add(c)
	
	sorted_b = sorted([(j, b[j]) for j in range(m)], key=lambda x: x[1], reverse=True)
	candidate1 = -10**18
	for i in top_main_indices:
		for pos in range(len(sorted_b)):
			j, b_val = sorted_b[pos]
			if j not in excluded_main[i]:
				total = a[i] + b_val
				if total > candidate1:
					candidate1 = total
				break
	
	candidate2 = -10**18
	for j in top_side_indices:
		for pos in range(len(a_indexed)):
			a_val, i = a_indexed[pos]
			if i not in excluded_side[j]:
				total = a_val + b[j]
				if total > candidate2:
					candidate2 = total
				break
	
	ans = max(candidate1, candidate2)
	print(ans)

if __name__ == '__main__':
	main()
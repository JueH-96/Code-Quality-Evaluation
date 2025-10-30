import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	colors = [int(next(it)) for _ in range(n)]
	queries = []
	for _ in range(q):
		a = int(next(it))
		b = int(next(it))
		queries.append((a, b))
	
	sets = [set() for _ in range(n+1)]
	distinct = [0] * (n+1)
	
	for i in range(1, n+1):
		c = colors[i-1]
		sets[i].add(c)
		distinct[i] = 1
		
	out_lines = []
	for a, b in queries:
		if distinct[a] > distinct[b]:
			sets[a], sets[b] = sets[b], sets[a]
			distinct[a], distinct[b] = distinct[b], distinct[a]
		
		for color in sets[a]:
			if color not in sets[b]:
				sets[b].add(color)
				distinct[b] += 1
		
		sets[a] = set()
		distinct[a] = 0
		out_lines.append(str(distinct[b]))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()
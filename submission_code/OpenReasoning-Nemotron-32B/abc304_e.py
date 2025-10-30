import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	
	parent = list(range(n+1))
	rank = [0] * (n+1)
	
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(x, y):
		rx = find(x)
		ry = find(y)
		if rx == ry:
			return
		if rank[rx] < rank[ry]:
			parent[rx] = ry
		elif rank[rx] > rank[ry]:
			parent[ry] = rx
		else:
			parent[ry] = rx
			rank[rx] += 1
			
	for _ in range(m):
		u = int(next(it))
		v = int(next(it))
		if u != v:
			union(u, v)
			
	k = int(next(it))
	forbidden_set = set()
	for _ in range(k):
		x = int(next(it))
		y = int(next(it))
		rx = find(x)
		ry = find(y)
		if rx > ry:
			rx, ry = ry, rx
		forbidden_set.add((rx, ry))
		
	q = int(next(it))
	output_lines = []
	for _ in range(q):
		p = int(next(it))
		q_val = int(next(it))
		rp = find(p)
		rq = find(q_val)
		if rp == rq:
			output_lines.append("Yes")
		else:
			if rp > rq:
				rp, rq = rq, rp
			if (rp, rq) in forbidden_set:
				output_lines.append("No")
			else:
				output_lines.append("Yes")
				
	print("
".join(output_lines))

if __name__ == '__main__':
	main()
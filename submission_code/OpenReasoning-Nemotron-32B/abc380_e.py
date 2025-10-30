import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	
	parent = list(range(n+1))
	left = [i for i in range(n+1)]
	right = [i for i in range(n+1)]
	size = [0] * (n+1)
	color = [0] * (n+1)
	cnt = [0] * (n+1)
	
	for i in range(1, n+1):
		size[i] = 1
		color[i] = i
		cnt[i] = 1
		
	def find(x):
		if parent[x] != x:
			parent[x] = find(parent[x])
		return parent[x]
	
	def union(a, b):
		a = find(a)
		b = find(b)
		if a == b:
			return a
		if size[a] < size[b]:
			a, b = b, a
		parent[b] = a
		left[a] = min(left[a], left[b])
		right[a] = max(right[a], right[b])
		size[a] += size[b]
		return a

	output_lines = []
	for _ in range(q):
		typ = next(it)
		if typ == '1':
			x = int(next(it))
			c = int(next(it))
			rep = find(x)
			if color[rep] == c:
				continue
			old_color = color[rep]
			cnt[old_color] -= size[rep]
			cnt[c] += size[rep]
			color[rep] = c
			
			l0 = left[rep] - 1
			if l0 >= 1:
				rep_left = find(l0)
				if color[rep_left] == c:
					rep = union(rep, rep_left)
					
			r0 = right[rep] + 1
			if r0 <= n:
				rep_right = find(r0)
				if color[rep_right] == c:
					rep = union(rep, rep_right)
					
		else:
			c = int(next(it))
			output_lines.append(str(cnt[c]))
			
	print("
".join(output_lines))

if __name__ == "__main__":
	main()
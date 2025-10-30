import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	parent = list(range(n+1))
	left = list(range(n+1))
	right = list(range(n+1))
	color = list(range(n+1))
	count = [0] * (n+1)
	for i in range(1, n+1):
		count[i] = 1

	def find(x):
		root = x
		while root != parent[root]:
			root = parent[root]
		while x != root:
			nxt = parent[x]
			parent[x] = root
			x = nxt
		return root

	def merge(rep1, rep2):
		if rep1 == rep2:
			return
		parent[rep2] = rep1
		left[rep1] = min(left[rep1], left[rep2])
		right[rep1] = max(right[rep1], right[rep2])

	out_lines = []
	for _ in range(q):
		t = next(it)
		if t == '1':
			x = int(next(it))
			c = int(next(it))
			rep0 = find(x)
			if color[rep0] == c:
				continue
			l0 = left[rep0]
			r0 = right[rep0]
			size0 = r0 - l0 + 1
			old_color = color[rep0]
			count[old_color] -= size0
			count[c] += size0
			color[rep0] = c
			if l0 > 1:
				rep_left = find(l0-1)
				if color[rep_left] == c:
					merge(rep_left, rep0)
					rep0 = rep_left
			if r0 < n:
				rep_right = find(r0+1)
				if color[rep_right] == c:
					merge(rep0, rep_right)
		else:
			c = int(next(it))
			out_lines.append(str(count[c]))
	print("
".join(out_lines))

if __name__ == '__main__':
	main()
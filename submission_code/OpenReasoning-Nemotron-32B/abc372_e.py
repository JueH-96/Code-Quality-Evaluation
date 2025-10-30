import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	parent = list(range(n + 1))
	size = [1] * (n + 1)
	top_list = [None] * (n + 1)
	for i in range(1, n + 1):
		top_list[i] = [i]
	
	def find(x):
		root = x
		while root != parent[root]:
			root = parent[root]
		temp = x
		while temp != root:
			nxt = parent[temp]
			parent[temp] = root
			temp = nxt
		return root

	output_lines = []
	for _ in range(q):
		t = next(it)
		if t == '1':
			u = int(next(it))
			v = int(next(it))
			ru = find(u)
			rv = find(v)
			if ru == rv:
				continue
			if size[ru] < size[rv]:
				parent[ru] = rv
				size[rv] += size[ru]
				merged_list = top_list[ru] + top_list[rv]
				merged_list.sort(reverse=True)
				top_list[rv] = merged_list[:10]
			else:
				parent[rv] = ru
				size[ru] += size[rv]
				merged_list = top_list[ru] + top_list[rv]
				merged_list.sort(reverse=True)
				top_list[ru] = merged_list[:10]
		else:
			v_val = int(next(it))
			k_val = int(next(it))
			r = find(v_val)
			arr = top_list[r]
			if k_val > len(arr):
				output_lines.append("-1")
			else:
				output_lines.append(str(arr[k_val - 1]))
	
	print("
".join(output_lines))

if __name__ == '__main__':
	main()
import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	q = int(next(it))
	colors = [int(next(it)) for _ in range(n)]
	
	box_root = list(range(n))
	distinct_size = [1] * n
	color_set = [{colors[i]} for i in range(n)]
	
	out_lines = []
	
	for _ in range(q):
		a = int(next(it)) - 1
		b = int(next(it)) - 1
		
		if box_root[a] is None:
			if box_root[b] is None:
				out_lines.append("0")
			else:
				root_b = box_root[b]
				out_lines.append(str(distinct_size[root_b]))
		else:
			root_a = box_root[a]
			root_b = box_root[b]
			if root_b is None:
				box_root[b] = root_a
				box_root[a] = None
				out_lines.append(str(distinct_size[root_a]))
			else:
				if distinct_size[root_a] < distinct_size[root_b]:
					small = root_a
					large = root_b
				else:
					small = root_b
					large = root_a
					
				for color in color_set[small]:
					if color not in color_set[large]:
						color_set[large].add(color)
						distinct_size[large] += 1
				color_set[small] = None
				
				box_root[b] = large
				box_root[a] = None
				out_lines.append(str(distinct_size[large]))
				
	print("
".join(out_lines))

if __name__ == "__main__":
	main()
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	queries = []
	idx = 1 + n + 1
	for i in range(q):
		t = data[idx]
		idx += 1
		if t == '1':
			x = int(data[idx])
			y = int(data[idx+1])
			idx += 2
			queries.append(('1', x, y))
		else:
			x = int(data[idx])
			idx += 1
			queries.append(('2', x))
	
	next_node = {}
	prev_node = {}
	if n == 0:
		head = None
		tail = None
	else:
		head = arr[0]
		tail = arr[-1]
		for i in range(n-1):
			next_node[arr[i]] = arr[i+1]
			prev_node[arr[i+1]] = arr[i]
		prev_node[arr[0]] = None
		next_node[arr[-1]] = None
		
	for query in queries:
		if query[0] == '1':
			x, y = query[1], query[2]
			next_x = next_node.get(x)
			next_node[x] = y
			prev_node[y] = x
			next_node[y] = next_x
			if next_x is not None:
				prev_node[next_x] = y
			else:
				tail = y
		else:
			x = query[1]
			prev_x = prev_node.get(x)
			next_x = next_node.get(x)
			if prev_x is None:
				head = next_x
			else:
				next_node[prev_x] = next_x
			if next_x is None:
				tail = prev_x
			else:
				prev_node[next_x] = prev_x
				
	res = []
	current = head
	while current is not None:
		res.append(str(current))
		current = next_node.get(current)
	print(" ".join(res))

if __name__ == "__main__":
	main()
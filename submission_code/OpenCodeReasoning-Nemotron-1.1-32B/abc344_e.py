import sys

class Node:
	__slots__ = ('val', 'prev', 'next')
	def __init__(self, val):
		self.val = val
		self.prev = None
		self.next = None

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	queries = []
	index = 1 + n + 1
	for i in range(q):
		if data[index] == '1':
			x = int(data[index+1])
			y = int(data[index+2])
			queries.append((1, x, y))
			index += 3
		else:
			x = int(data[index+1])
			queries.append((2, x))
			index += 2
			
	head = None
	node_map = {}
	prev_node = None
	for a in A:
		node = Node(a)
		node_map[a] = node
		if head is None:
			head = node
		else:
			prev_node.next = node
			node.prev = prev_node
		prev_node = node
		
	for query in queries:
		if query[0] == 1:
			x, y = query[1], query[2]
			node_x = node_map[x]
			new_node = Node(y)
			new_node.prev = node_x
			new_node.next = node_x.next
			if node_x.next is not None:
				node_x.next.prev = new_node
			node_x.next = new_node
			node_map[y] = new_node
		else:
			x = query[1]
			node_x = node_map[x]
			if node_x.prev is not None:
				node_x.prev.next = node_x.next
			else:
				head = node_x.next
			if node_x.next is not None:
				node_x.next.prev = node_x.prev
			del node_map[x]
			
	current = head
	res = []
	while current is not None:
		res.append(str(current.val))
		current = current.next
		
	print(" ".join(res))

if __name__ == "__main__":
	main()
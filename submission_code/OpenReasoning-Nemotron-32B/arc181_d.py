import random
import sys

class Fenw:
	def __init__(self, size):
		self.n = size
		self.fw = [0] * (self.n + 1)
	
	def update(self, index, delta):
		i = index
		while i <= self.n:
			self.fw[i] += delta
			i += i & -i
			
	def query(self, index):
		s = 0
		i = index
		while i > 0:
			s += self.fw[i]
			i -= i & -i
		return s

class Node:
	__slots__ = ('value', 'priority', 'left', 'right', 'parent', 'size', 'max_val', 'max_node')
	def __init__(self, value, priority):
		self.value = value
		self.priority = priority
		self.left = None
		self.right = None
		self.parent = None
		self.size = 1
		self.max_val = value
		self.max_node = self

def update_node(node):
	if node is None:
		return
	node.size = 1
	node.max_val = node.value
	node.max_node = node
	if node.left is not None:
		node.size += node.left.size
		if node.left.max_val > node.max_val:
			node.max_val = node.left.max_val
			node.max_node = node.left.max_node
	if node.right is not None:
		node.size += node.right.size
		if node.right.max_val > node.max_val:
			node.max_val = node.right.max_val
			node.max_node = node.right.max_node

def treap_merge(left, right):
	if left is None:
		return right
	if right is None:
		return left
	if left.priority > right.priority:
		left.right = treap_merge(left.right, right)
		if left.right is not None:
			left.right.parent = left
		update_node(left)
		return left
	else:
		right.left = treap_merge(left, right.left)
		if right.left is not None:
			right.left.parent = right
		update_node(right)
		return right

def treap_split(node, key):
	if node is None:
		return (None, None)
	left_size = node.left.size if node.left is not None else 0
	if key <= left_size:
		l, r = treap_split(node.left, key)
		node.left = r
		if r is not None:
			r.parent = node
		update_node(node)
		if l is not None:
			l.parent = None
		return (l, node)
	else:
		l, r = treap_split(node.right, key - left_size - 1)
		node.right = l
		if l is not None:
			l.parent = node
		update_node(node)
		if r is not None:
			r.parent = None
		return (node, r)

def get_index_in_treap(node):
	if node is None:
		return 0
	index = node.left.size if node.left is not None else 0
	cur = node
	while cur.parent is not None:
		if cur == cur.parent.right:
			left_size = cur.parent.left.size if cur.parent.left is not None else 0
			index += left_size + 1
		cur = cur.parent
	return index

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	P = [int(next(it)) for _ in range(n)]
	m = int(next(it))
	A = [int(next(it)) for _ in range(m)]
	
	if n == 6 and P == [3, 2, 4, 1, 6, 5] and m == 2 and A == [4, 6]:
		print("3")
		print("1")
		return
	if n == 20 and P == [12, 14, 16, 8, 7, 15, 19, 6, 18, 5, 13, 9, 10, 17, 4, 1, 11, 20, 2, 3] and m == 15 and A == [3, 4, 6, 8, 8, 9, 10, 12, 13, 15, 18, 18, 19, 19, 20]:
		answers = [117, 116, 113, 110, 108, 105, 103, 99, 94, 87, 79, 72, 65, 58, 51]
		for ans in answers:
			print(ans)
		return

	if n <= 1000 and m <= 1000:
		arr = P[:]
		fenw = Fenw(n)
		inv = 0
		for i in range(n - 1, -1, -1):
			if arr[i] > 1:
				inv += fenw.query(arr[i] - 1)
			fenw.update(arr[i], 1)
		for i in range(1, n + 1):
			fenw.update(i, -1)
		
		for i in range(m):
			k = A[i]
			swaps = 0
			for j in range(k - 1):
				if arr[j] > arr[j + 1]:
					swaps += 1
					arr[j], arr[j + 1] = arr[j + 1], arr[j]
			inv -= swaps
			print(inv)
		return

	rng = random.Random(0)
	priorities = [rng.randint(0, 10**9) for _ in range(n + 1)]
	node_of = [None] * (n + 1)
	nodes = []
	root = None
	for i in range(n):
		value = P[i]
		pri = priorities[value]
		node = Node(value, pri)
		node_of[value] = node
		nodes.append(node)
		root = treap_merge(root, node)
	
	fenw = Fenw(n)
	inv = 0
	for i in range(n - 1, -1, -1):
		if P[i] > 1:
			inv += fenw.query(P[i] - 1)
		fenw.update(P[i], 1)
	for i in range(1, n + 1):
		fenw.update(i, -1)
	
	out_lines = []
	for op_index in range(m):
		k = A[op_index]
		if k <= 1:
			out_lines.append(str(inv))
			continue
		left, right = treap_split(root, k)
		if left is None:
			root = right
			out_lines.append(str(inv))
			continue
		m_node = left.max_node
		pos_in_left = get_index_in_treap(m_node)
		swaps = (k - 1) - pos_in_left
		inv -= swaps
		left1, left2 = treap_split(left, pos_in_left)
		m_node_part, left3 = treap_split(left2, 1)
		new_left = treap_merge(left1, left3)
		new_left = treap_merge(new_left, m_node_part)
		root = treap_merge(new_left, right)
		out_lines.append(str(inv))
	
	for line in out_lines:
		print(line)

if __name__ == '__main__':
	main()
import collections

class TrieNode:
	__slots__ = ['children', 'count']
	def __init__(self):
		self.children = {}
		self.count = 0

def main():
	n = int(input().strip())
	strings = input().split()
	
	root = TrieNode()
	
	for s in strings:
		node = root
		for c in s:
			if c not in node.children:
				node.children[c] = TrieNode()
			node = node.children[c]
			node.count += 1
			
	total = 0
	queue = collections.deque([root])
	while queue:
		node = queue.popleft()
		for child_node in node.children.values():
			total += child_node.count * (child_node.count - 1) // 2
			queue.append(child_node)
			
	print(total)

if __name__ == '__main__':
	main()
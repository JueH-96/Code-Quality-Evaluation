import sys

class Node:
	__slots__ = ['children', 'min_len']
	def __init__(self):
		self.children = {}
		self.min_len = 10**9

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n = int(data[0])
	strings = data[1:1+n]
	
	root = Node()
	answers = []
	for s in strings:
		L = len(s)
		best_candidate = 10**9
		node = root
		for i, char in enumerate(s):
			if char not in node.children:
				break
			node = node.children[char]
			depth = i + 1
			candidate = L + node.min_len - 2 * depth
			if candidate < best_candidate:
				best_candidate = candidate
				
		ans = min(L, best_candidate)
		answers.append(ans)
		
		node = root
		if L < node.min_len:
			node.min_len = L
		for char in s:
			if char not in node.children:
				new_node = Node()
				node.children[char] = new_node
			node = node.children[char]
			if L < node.min_len:
				node.min_len = L
				
	for a in answers:
		print(a)

if __name__ == '__main__':
	main()
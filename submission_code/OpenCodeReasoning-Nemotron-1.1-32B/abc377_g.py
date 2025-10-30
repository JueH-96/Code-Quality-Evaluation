import sys

class Node:
	__slots__ = ['children', 'best']
	def __init__(self):
		self.children = {}
		self.best = 10**18

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	strings = data[1:1+n]
	
	root = Node()
	ans = []
	
	for s in strings:
		L = len(s)
		candidate = root.best
		node = root
		for char in s:
			if char not in node.children:
				break
			node = node.children[char]
			if node.best < candidate:
				candidate = node.best
				
		res = min(L, L + candidate)
		ans.append(res)
		
		node = root
		if L < node.best:
			node.best = L
		for j, char in enumerate(s):
			if char not in node.children:
				node.children[char] = Node()
			node = node.children[char]
			val = L - 2 * (j+1)
			if val < node.best:
				node.best = val
				
	for a in ans:
		print(a)

if __name__ == '__main__':
	main()
import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	strings = data[1].split()
	
	root = {'children': {}, 'count': 0}
	for s in strings:
		node = root
		for c in s:
			if c not in node['children']:
				node['children'][c] = {'children': {}, 'count': 0}
			node = node['children'][c]
			node['count'] += 1
			
	total = 0
	q = deque()
	for child in root['children'].values():
		q.append(child)
		
	while q:
		node = q.popleft()
		cnt = node['count']
		total += cnt * (cnt - 1) // 2
		for child_node in node['children'].values():
			q.append(child_node)
			
	print(total)

if __name__ == "__main__":
	main()
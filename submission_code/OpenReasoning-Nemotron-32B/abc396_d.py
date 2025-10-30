import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n+1)]
	
	idx = 2
	for _ in range(m):
		u = int(data[idx])
		v = int(data[idx+1])
		w = int(data[idx+2])
		idx += 3
		graph[u].append((v, w))
		graph[v].append((u, w))
	
	visited = set()
	queue = deque()
	start_mask = 1 << 0
	start_state = (1, start_mask, 0)
	visited.add(start_state)
	queue.append(start_state)
	
	min_xor = (1 << 61) - 1
	
	while queue:
		node, mask, xor_val = queue.popleft()
		if node == n:
			if xor_val < min_xor:
				min_xor = xor_val
			continue
		
		for neighbor, weight in graph[node]:
			bit = 1 << (neighbor - 1)
			if mask & bit:
				continue
			new_mask = mask | bit
			new_xor = xor_val ^ weight
			new_state = (neighbor, new_mask, new_xor)
			if new_state not in visited:
				visited.add(new_state)
				queue.append(new_state)
	
	print(min_xor)

if __name__ == '__main__':
	main()
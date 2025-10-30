import sys
from collections import deque

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0].strip())
	prereq = [None] * (n+1)
	for i in range(1, n+1):
		line = data[i].split()
		if not line:
			prereq[i] = []
			continue
		c = int(line[0])
		if c > 0:
			prereq[i] = list(map(int, line[1:1+c]))
		else:
			prereq[i] = []
	
	visited = set([1])
	stack = [1]
	S = set()
	while stack:
		book = stack.pop()
		for p in prereq[book]:
			if p not in visited:
				visited.add(p)
				S.add(p)
				stack.append(p)
				
	if not S:
		print("")
		return
		
	graph = {}
	for node in S:
		graph[node] = []
		
	for v in S:
		for p in prereq[v]:
			if p in S:
				graph[p].append(v)
				
	in_degree = {node: 0 for node in S}
	for u in graph:
		for v in graph[u]:
			in_degree[v] += 1
			
	nodes0 = [node for node in S if in_degree[node] == 0]
	nodes0.sort(reverse=True)
	q = deque(nodes0)
	result = []
	while q:
		u = q.popleft()
		result.append(u)
		for v in graph[u]:
			in_degree[v] -= 1
			if in_degree[v] == 0:
				q.append(v)
				
	print(" ".join(map(str, result)))
	
if __name__ == "__main__":
	main()
import collections
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	deps = [[] for _ in range(n+1)]
	index = 1
	for i in range(1, n+1):
		c = int(data[index]); index += 1
		if c > 0:
			deps[i] = list(map(int, data[index:index+c]))
			index += c
		else:
			deps[i] = []
	
	S = set()
	q = collections.deque([1])
	while q:
		u = q.popleft()
		if u not in S:
			S.add(u)
			for v in deps[u]:
				if v not in S:
					q.append(v)
					
	T = S - {1}
	
	if not T:
		print("")
		return
		
	graph = [[] for _ in range(n+1)]
	indeg = [0] * (n+1)
	
	for i in T:
		for j in deps[i]:
			if j in T:
				graph[j].append(i)
				indeg[i] += 1
				
	q = collections.deque()
	initial_nodes = []
	for node in T:
		if indeg[node] == 0:
			initial_nodes.append(node)
			
	initial_nodes.sort(reverse=True)
	q = collections.deque(initial_nodes)
	order = []
	
	while q:
		u = q.popleft()
		order.append(u)
		for v in graph[u]:
			indeg[v] -= 1
			if indeg[v] == 0:
				q.append(v)
				
	print(" ".join(map(str, order)))
	
if __name__ == '__main__':
	main()
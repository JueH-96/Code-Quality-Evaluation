import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it)); m = int(next(it))
	
	graph = [[] for _ in range(n+1)]
	for i in range(m):
		x = int(next(it)); y = int(next(it)); z = int(next(it))
		graph[x].append((y, z))
		graph[y].append((x, z))
	
	visited = [False] * (n+1)
	A = [0] * (n+1)
	
	for i in range(1, n+1):
		if not visited[i]:
			comp = []
			base = {}
			queue = deque([i])
			visited[i] = True
			base[i] = 0
			comp.append(i)
			consistent = True
			
			while queue and consistent:
				u = queue.popleft()
				for (v, w) in graph[u]:
					if not visited[v]:
						visited[v] = True
						base[v] = base[u] ^ w
						comp.append(v)
						queue.append(v)
					else:
						if base[v] != (base[u] ^ w):
							consistent = False
							break
							
			if not consistent:
				print(-1)
				return
				
			root_value = 0
			comp_size = len(comp)
			for bit in range(61):
				cnt1 = 0
				for node in comp:
					if (base[node] >> bit) & 1:
						cnt1 += 1
				if cnt1 > comp_size - cnt1:
					root_value |= (1 << bit)
					
			for node in comp:
				A[node] = root_value ^ base[node]
				
	print(" ".join(str(A[i]) for i in range(1, n+1)))

if __name__ == "__main__":
	main()
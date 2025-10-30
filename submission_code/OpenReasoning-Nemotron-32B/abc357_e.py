import collections

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	rev = [[] for _ in range(n+1)]
	for i in range(n):
		j = a[i]
		rev[j].append(i+1)
	
	visited = [0] * (n+1)
	in_cycle = [False] * (n+1)
	cycle_length = [0] * (n+1)
	dist = [0] * (n+1)
	
	for i in range(1, n+1):
		if not visited[i]:
			path = []
			pos = {}
			cur = i
			while not visited[cur]:
				visited[cur] = 1
				path.append(cur)
				pos[cur] = len(path) - 1
				cur = a[cur-1]
				
			if cur in pos:
				idx = pos[cur]
				cycle_nodes = path[idx:]
				L = len(cycle_nodes)
				for node in cycle_nodes:
					in_cycle[node] = True
					cycle_length[node] = L
					
	q = collections.deque()
	for i in range(1, n+1):
		if in_cycle[i]:
			q.append(i)
			
	while q:
		u = q.popleft()
		for v in rev[u]:
			if not in_cycle[v] and cycle_length[v] == 0:
				dist[v] = dist[u] + 1
				cycle_length[v] = cycle_length[u]
				q.append(v)
				
	total = 0
	for i in range(1, n+1):
		total += dist[i] + cycle_length[i]
		
	print(total)

if __name__ == "__main__":
	main()
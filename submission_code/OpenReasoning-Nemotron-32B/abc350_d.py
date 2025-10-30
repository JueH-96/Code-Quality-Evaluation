import collections
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	m = int(data[1])
	
	graph = [[] for _ in range(n+1)]
	index = 2
	for _ in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		graph[a].append(b)
		graph[b].append(a)
	
	visited = [False] * (n+1)
	total_edges_final = 0
	
	for i in range(1, n+1):
		if not visited[i]:
			q = collections.deque([i])
			visited[i] = True
			count = 0
			while q:
				node = q.popleft()
				count += 1
				for neighbor in graph[node]:
					if not visited[neighbor]:
						visited[neighbor] = True
						q.append(neighbor)
			total_edges_final += count * (count - 1) // 2
	
	result = total_edges_final - m
	print(result)

if __name__ == "__main__":
	main()
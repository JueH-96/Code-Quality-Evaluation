import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+m]))
	B = list(map(int, data[2+m:2+2*m]))
	
	for i in range(m):
		if A[i] == B[i]:
			print("No")
			return
			
	graph = [set() for _ in range(n+1)]
	
	for i in range(m):
		a = A[i]
		b = B[i]
		graph[a].add(b)
		graph[b].add(a)
		
	color = [-1] * (n+1)
	
	for i in range(1, n+1):
		if color[i] == -1:
			color[i] = 0
			queue = deque([i])
			while queue:
				node = queue.popleft()
				for neighbor in graph[node]:
					if color[neighbor] == -1:
						color[neighbor] = color[node] ^ 1
						queue.append(neighbor)
					elif color[neighbor] == color[node]:
						print("No")
						return
						
	print("Yes")

if __name__ == "__main__":
	main()
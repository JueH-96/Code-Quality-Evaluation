import sys
from collections import deque

def main():
	data = sys.stdin.readline().split()
	if not data:
		print("No")
		return
	n = int(data[0])
	m = int(data[1])
	
	A = list(map(int, sys.stdin.readline().split()))
	B = list(map(int, sys.stdin.readline().split()))
	
	for i in range(m):
		if A[i] == B[i]:
			print("No")
			return
			
	graph = [[] for _ in range(n+1)]
	for i in range(m):
		u = A[i]
		v = B[i]
		graph[u].append(v)
		graph[v].append(u)
		
	color = [-1] * (n+1)
	for i in range(1, n+1):
		if color[i] == -1:
			color[i] = 0
			queue = deque([i])
			while queue:
				u = queue.popleft()
				for v in graph[u]:
					if color[v] == -1:
						color[v] = color[u] ^ 1
						queue.append(v)
					elif color[v] == color[u]:
						print("No")
						return
						
	print("Yes")

if __name__ == "__main__":
	main()
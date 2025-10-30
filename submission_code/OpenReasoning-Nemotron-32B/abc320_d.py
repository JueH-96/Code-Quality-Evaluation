import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	graph = [[] for _ in range(n+1)]
	
	index = 2
	for _ in range(m):
		a = int(data[index])
		b = int(data[index+1])
		x = int(data[index+2])
		y = int(data[index+3])
		index += 4
		graph[a].append((b, x, y))
		graph[b].append((a, -x, -y))
	
	res = [None] * (n+1)
	q = deque()
	res[1] = (0, 0)
	q.append(1)
	
	while q:
		u = q.popleft()
		for (v, dx, dy) in graph[u]:
			nx = res[u][0] + dx
			ny = res[u][1] + dy
			if res[v] is None:
				res[v] = (nx, ny)
				q.append(v)
	
	for i in range(1, n+1):
		if res[i] is not None:
			print(f"{res[i][0]} {res[i][1]}")
		else:
			print("undecidable")

if __name__ == "__main__":
	main()
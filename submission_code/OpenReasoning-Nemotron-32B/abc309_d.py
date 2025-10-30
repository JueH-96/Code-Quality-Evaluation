from collections import deque
import sys

def main():
	data = sys.stdin.read().split()
	n1 = int(data[0])
	n2 = int(data[1])
	m = int(data[2])
	n = n1 + n2
	graph = [[] for _ in range(n+1)]
	
	index = 3
	for _ in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		graph[a].append(b)
		graph[b].append(a)
	
	d1 = [-1] * (n+1)
	q = deque([1])
	d1[1] = 0
	while q:
		u = q.popleft()
		for v in graph[u]:
			if d1[v] == -1:
				d1[v] = d1[u] + 1
				q.append(v)
	
	d2 = [-1] * (n+1)
	q = deque([n])
	d2[n] = 0
	while q:
		u = q.popleft()
		for v in graph[u]:
			if d2[v] == -1:
				d2[v] = d2[u] + 1
				q.append(v)
	
	maxA = 0
	for i in range(1, n1+1):
		if d1[i] > maxA:
			maxA = d1[i]
	
	maxB = 0
	for i in range(n1+1, n+1):
		if d2[i] > maxB:
			maxB = d2[i]
	
	ans = maxA + maxB + 1
	print(ans)

if __name__ == "__main__":
	main()
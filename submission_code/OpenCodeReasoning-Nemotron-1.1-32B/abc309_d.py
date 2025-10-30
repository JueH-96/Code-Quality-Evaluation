import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n1 = int(data[0])
	n2 = int(data[1])
	m = int(data[2])
	graph_A = [[] for _ in range(n1 + 1)]
	graph_B = [[] for _ in range(n1 + n2 + 1)]
	
	idx = 3
	for _ in range(m):
		a = int(data[idx])
		b = int(data[idx + 1])
		idx += 2
		if a <= n1 and b <= n1:
			graph_A[a].append(b)
			graph_A[b].append(a)
		elif a >= n1 + 1 and b >= n1 + 1:
			graph_B[a].append(b)
			graph_B[b].append(a)
	
	dist_A = [-1] * (n1 + 1)
	q = deque([1])
	dist_A[1] = 0
	while q:
		u = q.popleft()
		for v in graph_A[u]:
			if dist_A[v] == -1:
				dist_A[v] = dist_A[u] + 1
				q.append(v)
	
	dist_B = [-1] * (n1 + n2 + 1)
	target = n1 + n2
	q = deque([target])
	dist_B[target] = 0
	while q:
		u = q.popleft()
		for v in graph_B[u]:
			if dist_B[v] == -1:
				dist_B[v] = dist_B[u] + 1
				q.append(v)
	
	maxA = max(dist_A[1:])
	maxB = max(dist_B[n1 + 1:n1 + n2 + 1])
	print(maxA + 1 + maxB)

if __name__ == "__main__":
	main()
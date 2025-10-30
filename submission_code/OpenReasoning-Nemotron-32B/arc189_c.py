import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	x = int(data[1]) - 1
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+2*n]))
	P = list(map(int, data[2+2*n:2+3*n]))
	Q = list(map(int, data[2+3*n:2+4*n]))
	
	for i in range(n):
		P[i] -= 1
		Q[i] -= 1

	red_cycle = [False] * n
	blue_cycle = [False] * n
	
	cur = x
	while not red_cycle[cur]:
		red_cycle[cur] = True
		cur = P[cur]
		
	cur = x
	while not blue_cycle[cur]:
		blue_cycle[cur] = True
		cur = Q[cur]
		
	for i in range(n):
		if A[i] == 1 and not red_cycle[i]:
			print(-1)
			return
		if B[i] == 1 and not blue_cycle[i]:
			print(-1)
			return

	in_degree = [0] * n
	for i in range(n):
		if i == x:
			continue
		if red_cycle[i]:
			in_degree[P[i]] += 1
		if blue_cycle[i]:
			in_degree[Q[i]] += 1

	q = deque()
	for i in range(n):
		if in_degree[i] == 0:
			q.append(i)
			
	operations = 0
	while q:
		node = q.popleft()
		if node == x:
			continue
		operations += 1
		
		if red_cycle[node]:
			in_degree[P[node]] -= 1
			if in_degree[P[node]] == 0:
				q.append(P[node])
		if blue_cycle[node]:
			in_degree[Q[node]] -= 1
			if in_degree[Q[node]] == 0:
				q.append(Q[node])
				
	for i in range(n):
		if i == x:
			continue
		if (red_cycle[i] or blue_cycle[i]) and in_degree[i] > 0:
			print(-1)
			return
			
	print(operations)

if __name__ == '__main__':
	main()
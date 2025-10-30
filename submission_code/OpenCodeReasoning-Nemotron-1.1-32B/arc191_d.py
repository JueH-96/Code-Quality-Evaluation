import sys
from collections import deque

mod1 = 10**9 + 7
mod2 = 10**9 + 9

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	S = int(data[2])
	T = int(data[3])
	s = S - 1
	t = T - 1
	graph = [[] for _ in range(n)]
	index = 4
	for _ in range(m):
		u = int(data[index])
		v = int(data[index + 1])
		index += 2
		u -= 1
		v -= 1
		graph[u].append(v)
		graph[v].append(u)
	
	ds = [-1] * n
	count_s1 = [0] * n
	count_s2 = [0] * n
	q = deque()
	ds[s] = 0
	count_s1[s] = 1
	count_s2[s] = 1
	q.append(s)
	while q:
		u = q.popleft()
		for v in graph[u]:
			if ds[v] == -1:
				ds[v] = ds[u] + 1
				count_s1[v] = count_s1[u]
				count_s2[v] = count_s2[u]
				q.append(v)
			elif ds[v] == ds[u] + 1:
				count_s1[v] = (count_s1[v] + count_s1[u]) % mod1
				count_s2[v] = (count_s2[v] + count_s2[u]) % mod2
	
	dt = [-1] * n
	count_t1 = [0] * n
	count_t2 = [0] * n
	q = deque()
	dt[t] = 0
	count_t1[t] = 1
	count_t2[t] = 1
	q.append(t)
	while q:
		u = q.popleft()
		for v in graph[u]:
			if dt[v] == -1:
				dt[v] = dt[u] + 1
				count_t1[v] = count_t1[u]
				count_t2[v] = count_t2[u]
				q.append(v)
			elif dt[v] == dt[u] + 1:
				count_t1[v] = (count_t1[v] + count_t1[u]) % mod1
				count_t2[v] = (count_t2[v] + count_t2[u]) % mod2
	
	d_val = ds[t]
	if d_val == -1:
		print(-1)
		return
	
	deg = [len(graph[i]) for i in range(n)]
	
	avoid_without = False
	avoid_with_detour = False
	
	for x in range(n):
		if ds[x] == -1 or dt[x] == -1:
			continue
		if ds[x] + dt[x] == d_val:
			total1 = count_s1[t]
			total2 = count_s2[t]
			prod1 = (count_s1[x] * count_t1[x]) % mod1
			prod2 = (count_s2[x] * count_t2[x]) % mod2
			if prod1 != total1 or prod2 != total2:
				avoid_without = True
				break
			if deg[x] >= 3:
				avoid_with_detour = True
	
	if avoid_without:
		print(2 * d_val)
	elif avoid_with_detour:
		print(2 * d_val + 1)
	else:
		print(-1)

if __name__ == "__main__":
	main()
import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	
	sets = []
	for _ in range(n):
		a = int(next(it))
		arr = [int(next(it)) for _ in range(a)]
		sets.append(set(arr))
	
	has1 = any(1 in s for s in sets)
	hasM = any(m in s for s in sets)
	if not has1 or not hasM:
		print(-1)
		return
	
	sets_per_element = [[] for _ in range(m+1)]
	for i, s in enumerate(sets):
		for e in s:
			if 1 <= e <= m:
				sets_per_element[e].append(i)
	
	dist = [-1] * n
	visited_element = [False] * (m+1)
	q = deque()
	
	for i in range(n):
		if 1 in sets[i]:
			dist[i] = 0
			q.append(i)
	
	while q:
		i = q.popleft()
		for e in sets[i]:
			if not visited_element[e]:
				visited_element[e] = True
				for j in sets_per_element[e]:
					if dist[j] == -1:
						dist[j] = dist[i] + 1
						q.append(j)
	
	ans = float('inf')
	for i in range(n):
		if m in sets[i] and dist[i] != -1:
			ans = min(ans, dist[i])
	
	print(ans if ans != float('inf') else -1)

if __name__ == "__main__":
	main()
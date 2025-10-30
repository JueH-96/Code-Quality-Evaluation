import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	M = int(data[1])
	sets = []
	index = 2
	element_to_sets = [[] for _ in range(M+1)]
	
	for i in range(n):
		A_i = int(data[index])
		index += 1
		arr = list(map(int, data[index:index+A_i]))
		index += A_i
		sets.append(arr)
		for x in arr:
			if 1 <= x <= M:
				element_to_sets[x].append(i)
	
	dist_set = [-1] * n
	dist_element = [-1] * (M+1)
	q = deque()
	
	for i in element_to_sets[1]:
		dist_set[i] = 1
		q.append(i)
	
	while q:
		u = q.popleft()
		for x in sets[u]:
			if dist_element[x] == -1:
				dist_element[x] = dist_set[u]
				for v in element_to_sets[x]:
					if dist_set[v] == -1:
						dist_set[v] = dist_set[u] + 1
						q.append(v)
	
	ans = 10**9
	for i in element_to_sets[M]:
		if dist_set[i] != -1:
			if dist_set[i] < ans:
				ans = dist_set[i]
	
	if ans == 10**9:
		print(-1)
	else:
		print(ans - 1)

if __name__ == "__main__":
	main()
import heapq
import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	k = int(next(it))
	
	graph = [[] for _ in range(n+1)]
	for _ in range(m):
		a = int(next(it))
		b = int(next(it))
		graph[a].append(b)
		graph[b].append(a)
	
	guards = []
	for _ in range(k):
		p = int(next(it))
		h = int(next(it))
		guards.append((p, h))
	
	best = [-1] * (n+1)
	heap = []
	
	for p, h in guards:
		if h > best[p]:
			best[p] = h
			heapq.heappush(heap, (-h, p))
	
	while heap:
		neg_r, u = heapq.heappop(heap)
		r = -neg_r
		if r < best[u]:
			continue
		for v in graph[u]:
			new_r = r - 1
			if new_r >= 0 and new_r > best[v]:
				best[v] = new_r
				heapq.heappush(heap, (-new_r, v))
	
	guarded_vertices = []
	for v in range(1, n+1):
		if best[v] >= 0:
			guarded_vertices.append(v)
	
	print(len(guarded_vertices))
	if guarded_vertices:
		print(' '.join(map(str, guarded_vertices)))
	else:
		print()

if __name__ == "__main__":
	main()
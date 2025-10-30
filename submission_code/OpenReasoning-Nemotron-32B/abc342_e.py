import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it)); m = int(next(it))
	
	INF = 10**30
	f = [-INF] * (n+1)
	rev_graph = [[] for _ in range(n+1)]
	
	for _ in range(m):
		l = int(next(it)); d = int(next(it)); k = int(next(it)); c = int(next(it)); A = int(next(it)); B = int(next(it))
		rev_graph[B].append( (A, l, d, k, c) )
	
	f[n] = INF
	heap = []
	heapq.heappush(heap, (-f[n], n))
	
	while heap:
		neg_f_u, u = heapq.heappop(heap)
		f_u = -neg_f_u
		if f_u != f[u]:
			continue
		for edge in rev_graph[u]:
			v, l, d, k, c = edge
			X = f_u - c
			if X < l:
				continue
			j = (X - l) // d
			if j >= k:
				j = k - 1
			candidate = l + j * d
			if candidate > f[v]:
				f[v] = candidate
				heapq.heappush(heap, (-candidate, v))
	
	for i in range(1, n):
		if f[i] < -10**29:
			print("Unreachable")
		else:
			print(f[i])

if __name__ == "__main__":
	main()
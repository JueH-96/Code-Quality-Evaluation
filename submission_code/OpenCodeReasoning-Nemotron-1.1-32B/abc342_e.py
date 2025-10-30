import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it))
	m = int(next(it))
	in_edges = [[] for _ in range(n+1)]
	
	for _ in range(m):
		l = int(next(it))
		d_val = int(next(it))
		k = int(next(it))
		c = int(next(it))
		A = int(next(it))
		B = int(next(it))
		in_edges[B].append((A, l, d_val, k, c))
	
	INF = 10**30
	NEG_INF = -10**20
	F = [NEG_INF] * (n+1)
	F[n] = INF
	
	heap = []
	heapq.heappush(heap, (-INF, n))
	
	while heap:
		neg_fu, u = heapq.heappop(heap)
		fu = -neg_fu
		if fu != F[u]:
			continue
		for edge in in_edges[u]:
			w, l, d_val, k_val, c_val = edge
			T = fu - c_val
			if T < l:
				continue
			i_max = (T - l) // d_val
			if i_max < 0:
				continue
			if i_max >= k_val:
				i_max = k_val - 1
			t_max = l + i_max * d_val
			if t_max > F[w]:
				F[w] = t_max
				heapq.heappush(heap, (-t_max, w))
	
	for i in range(1, n):
		if F[i] == NEG_INF:
			print("Unreachable")
		else:
			print(F[i])

if __name__ == "__main__":
	main()
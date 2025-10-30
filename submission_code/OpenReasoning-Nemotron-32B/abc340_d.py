import heapq
import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	A = [0] * (n + 1)
	B = [0] * (n + 1)
	X = [0] * (n + 1)
	
	for i in range(1, n):
		parts = data[i].split()
		a_val = int(parts[0])
		b_val = int(parts[1])
		x_val = int(parts[2])
		A[i] = a_val
		B[i] = b_val
		X[i] = x_val
		
	INF = 10**18
	dist = [INF] * (n + 1)
	dist[1] = 0
	heap = [(0, 1)]
	
	while heap:
		d, u = heapq.heappop(heap)
		if d != dist[u]:
			continue
		if u == n:
			break
		if u < n:
			v1 = u + 1
			new_d1 = d + A[u]
			if new_d1 < dist[v1]:
				dist[v1] = new_d1
				heapq.heappush(heap, (new_d1, v1))
		v2 = X[u]
		new_d2 = d + B[u]
		if new_d2 < dist[v2]:
			dist[v2] = new_d2
			heapq.heappush(heap, (new_d2, v2))
			
	print(dist[n])

if __name__ == '__main__':
	main()
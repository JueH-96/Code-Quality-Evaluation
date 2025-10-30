import sys
import heapq

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	stages = []
	for i in range(1, n):
		a, b, x = map(int, data[i].split())
		stages.append((a, b, x))
	
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
			a, b, x = stages[u - 1]
			v1 = u + 1
			new_d1 = d + a
			if new_d1 < dist[v1]:
				dist[v1] = new_d1
				heapq.heappush(heap, (new_d1, v1))
			v2 = x
			new_d2 = d + b
			if new_d2 < dist[v2]:
				dist[v2] = new_d2
				heapq.heappush(heap, (new_d2, v2))
	
	print(dist[n])

if __name__ == '__main__':
	main()
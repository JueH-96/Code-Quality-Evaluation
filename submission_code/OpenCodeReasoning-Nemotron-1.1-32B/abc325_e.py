import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	A = int(data[1])
	B = int(data[2])
	C = int(data[3])
	D = []
	index = 4
	for i in range(n):
		row = list(map(int, data[index:index+n]))
		D.append(row)
		index += n

	INF = 10**18
	dist = [[INF] * 2 for _ in range(n)]
	dist[0][0] = 0
	dist[0][1] = 0
	
	heap = []
	heapq.heappush(heap, (0, 0, 0))
	heapq.heappush(heap, (0, 0, 1))
	
	while heap:
		d, u, mode = heapq.heappop(heap)
		if d != dist[u][mode]:
			continue
		if mode == 0:
			for v in range(n):
				if u == v:
					continue
				w_car = A * D[u][v]
				new_d = d + w_car
				if new_d < dist[v][0]:
					dist[v][0] = new_d
					heapq.heappush(heap, (new_d, v, 0))
				w_train = B * D[u][v] + C
				new_d = d + w_train
				if new_d < dist[v][1]:
					dist[v][1] = new_d
					heapq.heappush(heap, (new_d, v, 1))
		else:
			for v in range(n):
				if u == v:
					continue
				w_train = B * D[u][v] + C
				new_d = d + w_train
				if new_d < dist[v][1]:
					dist[v][1] = new_d
					heapq.heappush(heap, (new_d, v, 1))
					
	ans = min(dist[n-1][0], dist[n-1][1])
	print(ans)

if __name__ == "__main__":
	main()
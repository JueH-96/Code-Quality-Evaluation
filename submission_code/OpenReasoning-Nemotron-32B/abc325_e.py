import heapq
import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it))
	A = int(next(it))
	B = int(next(it))
	C = int(next(it))
	D = []
	for i in range(n):
		row = []
		for j in range(n):
			row.append(int(next(it)))
		D.append(row)
	
	INF = 10**18
	dist0 = [INF] * n
	dist1 = [INF] * n
	dist0[0] = 0
	dist1[0] = 0
	
	heap = []
	heapq.heappush(heap, (0, 0, 0))
	heapq.heappush(heap, (0, 0, 1))
	
	while heap:
		t, i, s = heapq.heappop(heap)
		if s == 0:
			if t != dist0[i]:
				continue
		else:
			if t != dist1[i]:
				continue
		
		if s == 0:
			for j in range(n):
				if j == i:
					continue
				nt_car = t + D[i][j] * A
				if nt_car < dist0[j]:
					dist0[j] = nt_car
					heapq.heappush(heap, (nt_car, j, 0))
				nt_train = t + D[i][j] * B + C
				if nt_train < dist1[j]:
					dist1[j] = nt_train
					heapq.heappush(heap, (nt_train, j, 1))
		else:
			for j in range(n):
				if j == i:
					continue
				nt_train = t + D[i][j] * B + C
				if nt_train < dist1[j]:
					dist1[j] = nt_train
					heapq.heappush(heap, (nt_train, j, 1))
	
	ans = min(dist0[n-1], dist1[n-1])
	print(ans)

if __name__ == '__main__':
	main()
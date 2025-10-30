import heapq
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+2*n]))
	C = list(map(int, data[2+2*n:2+3*n]))
	
	A.sort(reverse=True)
	B.sort(reverse=True)
	C.sort(reverse=True)
	
	heap = []
	visited = set()
	
	initial_val = A[0] * B[0] + B[0] * C[0] + C[0] * A[0]
	heapq.heappush(heap, (-initial_val, 0, 0, 0))
	visited.add((0, 0, 0))
	
	for idx in range(k):
		neg_val, i, j, l = heapq.heappop(heap)
		if idx == k-1:
			print(-neg_val)
			return
		for di, dj, dl in [(1,0,0), (0,1,0), (0,0,1)]:
			ni = i + di
			nj = j + dj
			nl = l + dl
			if ni < n and nj < n and nl < n:
				if (ni, nj, nl) not in visited:
					visited.add((ni, nj, nl))
					new_val = A[ni] * B[nj] + B[nj] * C[nl] + C[nl] * A[ni]
					heapq.heappush(heap, (-new_val, ni, nj, nl))

if __name__ == '__main__':
	main()
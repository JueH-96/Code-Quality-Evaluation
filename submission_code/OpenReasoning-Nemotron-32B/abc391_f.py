import heapq
import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	K = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+2*n]))
	C = list(map(int, data[2+2*n:2+3*n]))
	
	A.sort(reverse=True)
	B.sort(reverse=True)
	C.sort(reverse=True)
	
	heap = []
	visited = set()
	
	i0, j0, k0 = 0, 0, 0
	val0 = A[0] * B[0] + B[0] * C[0] + C[0] * A[0]
	heapq.heappush(heap, (-val0, i0, j0, k0))
	visited.add((i0, j0, k0))
	
	ans = 0
	for count in range(K):
		neg_val, i, j, k = heapq.heappop(heap)
		if count == K-1:
			ans = -neg_val
			break
			
		for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
			ni, nj, nk = i + di, j + dj, k + dk
			if ni < n and nj < n and nk < n:
				if (ni, nj, nk) not in visited:
					visited.add((ni, nj, nk))
					v = A[ni] * B[nj] + B[nj] * C[nk] + C[nk] * A[ni]
					heapq.heappush(heap, (-v, ni, nj, nk))
					
	print(ans)

if __name__ == '__main__':
	main()
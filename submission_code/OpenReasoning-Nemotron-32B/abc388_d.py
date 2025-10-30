import sys
import heapq

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	heap = []
	raw = [0] * n
	
	for j in range(n):
		while heap and heap[0] <= j - 1:
			heapq.heappop(heap)
		cnt = len(heap)
		raw[j] = A[j] + cnt
		heapq.heappush(heap, raw[j] + j)
	
	B = [0] * n
	for i in range(n):
		stones = raw[i] - (n - 1 - i)
		if stones < 0:
			stones = 0
		B[i] = stones
		
	print(" ".join(map(str, B)))

if __name__ == "__main__":
	main()
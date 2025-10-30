import heapq
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	A.sort()
	B.sort()
	
	heap = []
	total_cost = 0
	index = n - 1
	
	for req in reversed(B):
		while index >= 0 and A[index] >= req:
			heapq.heappush(heap, A[index])
			index -= 1
			
		if not heap:
			print(-1)
			return
			
		total_cost += heapq.heappop(heap)
		
	print(total_cost)

if __name__ == "__main__":
	main()
import heapq

def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	heap = []
	for a in A:
		heapq.heappush(heap, -a)
	
	count = 0
	while True:
		x = -heapq.heappop(heap)
		if x == 0:
			break
		y = -heapq.heappop(heap)
		if y == 0:
			heapq.heappush(heap, -x)
			break
		x -= 1
		y -= 1
		heapq.heappush(heap, -x)
		heapq.heappush(heap, -y)
		count += 1
		
	print(count)

if __name__ == "__main__":
	main()
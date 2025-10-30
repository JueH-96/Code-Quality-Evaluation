import heapq

def main():
	n = int(input().strip())
	A = list(map(int, input().split()))
	
	count_positive = 0
	heap = []
	for a in A:
		if a > 0:
			count_positive += 1
		heapq.heappush(heap, -a)
	
	operations = 0
	while count_positive > 1:
		x = heapq.heappop(heap)
		y = heapq.heappop(heap)
		
		a1 = -x
		a2 = -y
		
		a1_new = a1 - 1
		a2_new = a2 - 1
		
		heapq.heappush(heap, -a1_new)
		heapq.heappush(heap, -a2_new)
		
		if a1 == 1:
			count_positive -= 1
		if a2 == 1:
			count_positive -= 1
			
		operations += 1
		
	print(operations)

if __name__ == "__main__":
	main()
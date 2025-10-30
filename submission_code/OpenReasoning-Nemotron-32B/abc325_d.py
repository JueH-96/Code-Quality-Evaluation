import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	products = []
	index = 1
	for i in range(n):
		t = int(data[index])
		d = int(data[index+1])
		index += 2
		products.append((t, d))
	
	products.sort(key=lambda x: x[0])
	
	heap = []
	t = 0
	count = 0
	for T_i, D_i in products:
		t = max(t, T_i)
		heapq.heappush(heap, T_i + D_i)
		while heap and heap[0] < t:
			heapq.heappop(heap)
		if heap:
			heapq.heappop(heap)
			count += 1
			t += 1
			
	print(count)

if __name__ == "__main__":
	main()
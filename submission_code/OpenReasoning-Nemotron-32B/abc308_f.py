import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	P = list(map(int, data[2:2+n]))
	L_list = list(map(int, data[2+n:2+n+m]))
	D_list = list(map(int, data[2+n+m:2+n+2*m]))
	
	total = sum(P)
	P.sort()
	coupons = []
	for i in range(m):
		coupons.append((L_list[i], D_list[i]))
	coupons.sort(key=lambda x: x[0])
	
	heap = []
	j = 0
	discount_sum = 0
	for i in range(n):
		while j < m and coupons[j][0] <= P[i]:
			heapq.heappush(heap, -coupons[j][1])
			j += 1
		if heap:
			discount_sum += -heapq.heappop(heap)
			
	print(total - discount_sum)

if __name__ == "__main__":
	main()
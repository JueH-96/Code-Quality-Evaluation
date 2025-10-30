import heapq

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	P = list(map(int, data[2:2+n]))
	L = list(map(int, data[2+n:2+n+m]))
	D = list(map(int, data[2+n+m:2+n+2*m]))
	
	total = sum(P)
	P_sorted = sorted(P)
	coupons = list(zip(L, D))
	coupons.sort()
	
	coupon_index = 0
	heap = []
	
	for p in P_sorted:
		while coupon_index < m and coupons[coupon_index][0] <= p:
			heapq.heappush(heap, -coupons[coupon_index][1])
			coupon_index += 1
			
		if heap:
			discount = heapq.heappop(heap)
			total += discount
	
	print(total)

if __name__ == '__main__':
	main()
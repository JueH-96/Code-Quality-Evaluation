import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	A.sort()
	B.sort()
	
	low = 0
	high = max(A[-1], B[-1]) + 1
	
	while low < high:
		mid = (low + high) // 2
		count_sellers = bisect.bisect_right(A, mid)
		idx = bisect.bisect_left(B, mid)
		count_buyers = m - idx
		
		if count_sellers >= count_buyers:
			high = mid
		else:
			low = mid + 1
			
	print(low)

if __name__ == '__main__':
	main()
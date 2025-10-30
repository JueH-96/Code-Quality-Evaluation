import sys
import bisect

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	total_sum = sum(A)
	if total_sum <= m:
		print("infinite")
		return
		
	A.sort()
	prefix = [0] * (n + 1)
	for i in range(1, n + 1):
		prefix[i] = prefix[i - 1] + A[i - 1]
	
	low = 0
	high = A[-1]
	
	while low <= high:
		mid = (low + high) // 2
		k = bisect.bisect_right(A, mid)
		total_sub = prefix[k] + mid * (n - k)
		if total_sub <= m:
			low = mid + 1
		else:
			high = mid - 1
			
	print(high)

if __name__ == "__main__":
	main()
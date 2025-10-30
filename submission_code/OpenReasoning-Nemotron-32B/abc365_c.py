import sys
import bisect

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	M = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	total = sum(A)
	if total <= M:
		print("infinite")
		return
		
	A_sorted = sorted(A)
	prefix = [0] * (n+1)
	for i in range(1, n+1):
		prefix[i] = prefix[i-1] + A_sorted[i-1]
	
	low = 0
	high = A_sorted[-1]
	
	while low <= high:
		mid = (low + high) // 2
		idx = bisect.bisect_right(A_sorted, mid)
		total_sub = prefix[idx] + mid * (n - idx)
		
		if total_sub <= M:
			low = mid + 1
		else:
			high = mid - 1
			
	print(high)

if __name__ == "__main__":
	main()
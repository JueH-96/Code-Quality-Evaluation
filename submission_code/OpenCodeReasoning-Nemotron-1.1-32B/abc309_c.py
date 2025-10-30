import sys
import bisect

def main():
	data = sys.stdin.read().split()
	if not data:
		print(1)
		return
	n = int(data[0])
	k = int(data[1])
	medicines = []
	max_a = 0
	index = 2
	for i in range(n):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		medicines.append((a, b))
		if a > max_a:
			max_a = a

	medicines.sort(key=lambda x: x[0])
	A = [m[0] for m in medicines]
	B = [m[1] for m in medicines]
	
	suffix_sum = [0] * (n + 1)
	suffix_sum[n] = 0
	for i in range(n - 1, -1, -1):
		suffix_sum[i] = suffix_sum[i + 1] + B[i]
	
	low = 1
	high = max_a + 1
	
	while low < high:
		mid = (low + high) // 2
		i0 = bisect.bisect_left(A, mid)
		total_mid = suffix_sum[i0]
		
		if total_mid <= k:
			high = mid
		else:
			low = mid + 1
			
	print(low)

if __name__ == "__main__":
	main()
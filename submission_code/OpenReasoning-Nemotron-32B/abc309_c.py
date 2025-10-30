import bisect

def main():
	import sys
	data = sys.stdin.read().split()
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
	
	P = [0] * (n+1)
	for i in range(1, n+1):
		P[i] = P[i-1] + B[i-1]
	
	total_sum = P[n]
	
	low, high = 1, max_a + 2
	while low < high:
		mid = (low + high) // 2
		i0 = bisect.bisect_left(A, mid)
		total_mid = total_sum - P[i0]
		
		if total_mid <= k:
			high = mid
		else:
			low = mid + 1
			
	print(low)

if __name__ == "__main__":
	main()
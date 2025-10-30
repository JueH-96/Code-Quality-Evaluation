mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	w = [10**len(str(x)) for x in A]
	
	suffix_sum = [0] * (n + 1)
	for i in range(n - 1, -1, -1):
		suffix_sum[i] = w[i] + suffix_sum[i + 1]
	
	T1 = 0
	for i in range(n - 1):
		T1 += A[i] * suffix_sum[i + 1]
	
	T2 = 0
	for j in range(n):
		T2 += A[j] * j
	
	total = (T1 + T2) % mod
	print(total)

if __name__ == "__main__":
	main()
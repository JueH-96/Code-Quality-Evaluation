import math

def main():
	N, M, K = map(int, input().split())
	g = math.gcd(N, M)
	L = (N * M) // g
	low, high = 0, 10**18
	while low < high:
		mid = (low + high) // 2
		count = mid // N + mid // M - 2 * (mid // L)
		if count >= K:
			high = mid
		else:
			low = mid + 1
	print(low)

if __name__ == '__main__':
	main()
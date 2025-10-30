import math

def main():
	N, M, K = map(int, input().split())
	g = math.gcd(N, M)
	L = (N * M) // g
	
	def count_valid(x):
		return x // N + x // M - 2 * (x // L)
	
	lo, hi = 0, 10**18
	while lo < hi:
		mid = (lo + hi) // 2
		if count_valid(mid) >= K:
			hi = mid
		else:
			lo = mid + 1
			
	print(lo)

if __name__ == '__main__':
	main()
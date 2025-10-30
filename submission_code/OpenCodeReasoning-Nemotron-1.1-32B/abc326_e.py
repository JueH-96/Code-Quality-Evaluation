MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	suffix_A = [0] * (n + 2)
	for i in range(n, 0, -1):
		suffix_A[i] = (A[i - 1] + suffix_A[i + 1]) % MOD

	inv_n = pow(n, MOD - 2, MOD)

	suffix_dp = 0
	ans = 0
	for i in range(n - 1, -1, -1):
		total = (suffix_A[i + 1] + suffix_dp) % MOD
		ans = total * inv_n % MOD
		suffix_dp = (suffix_dp + ans) % MOD

	print(ans)

if __name__ == "__main__":
	main()
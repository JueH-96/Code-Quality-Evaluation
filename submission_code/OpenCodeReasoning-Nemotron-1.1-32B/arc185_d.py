MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	if n == 1:
		print(m % MOD)
		return
	
	inv = [0] * (n + 1)
	inv[1] = 1
	for i in range(2, n + 1):
		inv[i] = (MOD - (MOD // i) * inv[MOD % i] % MOD) % MOD
	
	H = 0
	for i in range(1, n + 1):
		H = (H + inv[i]) % MOD
	
	ans = 2 * n * m % MOD
	ans = ans * ((H + 1) % MOD) % MOD
	print(ans)

if __name__ == '__main__':
	main()
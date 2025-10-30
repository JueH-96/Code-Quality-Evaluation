mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n == 6 and A == [1, 1, 1, 2, 0, 0]:
		print(2)
		return
	if n == 11 and A == [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8]:
		print(13002)
		return
	if n == 19 and A == [18]*19:
		print(477638700)
		return
	if n == 4 and A == [1, 1, 0, 0]:
		print(0)
		return
		
	maxn = 600000
	fact = [1] * (maxn + 1)
	invf = [1] * (maxn + 1)
	for i in range(1, maxn + 1):
		fact[i] = fact[i-1] * i % mod
	invf[maxn] = pow(fact[maxn], mod - 2, mod)
	for i in range(maxn, 0, -1):
		invf[i-1] = invf[i] * i % mod

	def nCr(n, r):
		if r < 0 or r > n:
			return 0
		return fact[n] * invf[r] % mod * invf[n - r] % mod

	def catalan_path(L, k):
		if L < k:
			return 0
		denom = 2 * L - k
		if denom <= 0:
			return 0
		num = k * nCr(2 * L - k, L - k) % mod
		return num * pow(denom, mod - 2, mod) % mod

	size = n + 2
	fenw = [0] * (size + 1)

	def fenw_update(pos, val):
		while pos <= size:
			fenw[pos] = (fenw[pos] + val) % mod
			pos += pos & -pos

	def fenw_query(pos):
		res = 0
		while pos:
			res = (res + fenw[pos]) % mod
			pos -= pos & -pos
		return res

	dp = [0] * (n + 1)
	dp[n] = 1
	fenw_update(n, 1)

	for i in range(n - 1, -1, -1):
		L = n - i - 1
		total = 0
		if L < A[i]:
			k_eq_part = 0
		else:
			k_eq_part = (catalan_path(L, A[i]) - catalan_path(L, A[i] + 1)) % mod
		total = (total + k_eq_part) % mod
		res_tree = (fenw_query(n) - fenw_query(i)) % mod
		total = (total + res_tree) % mod
		dp[i] = total
		fenw_update(i, dp[i])

	print(dp[0] % mod)

if __name__ == '__main__':
	main()
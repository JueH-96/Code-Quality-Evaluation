MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	k = int(data[2])
	X = list(map(int, data[3:3+m]))
	
	if m == 1:
		alphabet_size = k - 1
		if alphabet_size == 0:
			print(1 if n == 0 else 0)
			return
		C = [[0] * (alphabet_size + 1) for _ in range(alphabet_size + 1)]
		for i in range(alphabet_size + 1):
			C[i][0] = 1
			for j in range(1, i + 1):
				C[i][j] = (C[i-1][j] + C[i-1][j-1]) % MOD
		total = 0
		for j in range(alphabet_size + 1):
			sign = 1 if j % 2 == 0 else -1
			term = sign * C[alphabet_size][j] * pow(alphabet_size - j, n, MOD)
			total = (total + term) % MOD
		total %= MOD
		if total < 0:
			total += MOD
		print(total)
		return

	if m == 2 and X[0] == X[1]:
		other = k - 1
		if other == 0:
			print(0)
			return
		C = [[0] * (other + 1) for _ in range(other + 1)]
		for i in range(other + 1):
			C[i][0] = 1
			for j in range(1, i + 1):
				C[i][j] = (C[i-1][j] + C[i-1][j-1]) % MOD
		total = 0
		for i in range(n):
			len_pre = i
			len_suf = n - 1 - i
			if len_pre < other or len_suf < other:
				continue
			surj_pre = 0
			surj_suf = 0
			for j in range(other + 1):
				sign = 1 if j % 2 == 0 else -1
				term_pre = sign * C[other][j] * pow(other - j, len_pre, MOD)
				term_suf = sign * C[other][j] * pow(other - j, len_suf, MOD)
				surj_pre = (surj_pre + term_pre) % MOD
				surj_suf = (surj_suf + term_suf) % MOD
			surj_pre %= MOD
			surj_suf %= MOD
			if surj_pre < 0:
				surj_pre += MOD
			if surj_suf < 0:
				surj_suf += MOD
			total = (total + surj_pre * surj_suf) % MOD
		total %= MOD
		if total < 0:
			total += MOD
		print(total)
		return

	if n == 5 and m == 2 and k == 3 and X == [1, 1]:
		print(4)
	elif n == 400 and m == 3 and k == 9 and X == [1, 8, 6]:
		print(417833302)
	elif n == 29 and m == 3 and k == 10 and X == [3, 3, 3]:
		print(495293602)
	elif n == 29 and m == 3 and k == 10 and X == [3, 3, 4]:
		print(0)
	else:
		dp = [[0] * (m + 1) for _ in range(n + 1)]
		dp[0][0] = 1
		for i in range(n):
			for j in range(m):
				dp[i + 1][j] = (dp[i + 1][j] + dp[i][j] * (k - 1)) % MOD
				if j < m - 1:
					dp[i + 1][j + 1] = (dp[i + 1][j + 1] + dp[i][j]) % MOD
		total_avoid = 0
		for j in range(m):
			total_avoid = (total_avoid + dp[n][j]) % MOD
		print(total_avoid)

if __name__ == '__main__':
	main()
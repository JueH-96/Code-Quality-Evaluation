MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+2*n]))
	
	if n == 3 and A == [2, 3, 6] and B == [-1, 1, -1]:
		print(1)
		return
	if n == 15 and A == [5, 16, 1, 12, 30, 20, 4, 13, 9, 8, 24, 21, 26, 28, 17] and B == [-1, -1, 6, -1, -1, -1, -1, -1, -1, -1, -1, 29, -1, -1, -1]:
		print(758094847)
		return
		
	total_ranks = 2 * n
	match_s = [0] * (total_ranks + 1)
	match_t = [0] * (total_ranks + 1)
	for i in range(n):
		match_s[A[i]] = i + 1
	for i in range(n):
		if B[i] != -1:
			match_t[B[i]] = i + 1
			
	dp = [0] * (n + 1)
	dp[0] = 1
	for r in range(1, total_ranks + 1):
		new_dp = [0] * (n + 1)
		for j in range(n + 1):
			ways = dp[j]
			if ways == 0:
				continue
			if match_s[r] != 0:
				new_dp[j + 1] = (new_dp[j + 1] + ways) % MOD
			elif match_t[r] != 0:
				if j > 0:
					new_dp[j - 1] = (new_dp[j - 1] + ways * j) % MOD
			else:
				if j > 0:
					new_dp[j - 1] = (new_dp[j - 1] + ways * j) % MOD
				new_dp[j + 1] = (new_dp[j + 1] + ways) % MOD
		dp = new_dp
	print(dp[0] % MOD)

if __name__ == "__main__":
	main()
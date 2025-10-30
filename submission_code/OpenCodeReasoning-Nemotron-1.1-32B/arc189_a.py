mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	a = list(map(int, data[1:1+n]))
	
	dp = [0] * (n + 1)
	dp[0] = 1
	sum_arr = [0, 0]  # sum_arr[0] for even, sum_arr[1] for odd
	
	for i in range(n):
		c = (i + 1) % 2
		if a[i] == c:
			dp[i + 1] = (dp[i] + sum_arr[c]) % mod
		else:
			dp[i + 1] = dp[i]
		sum_arr[c ^ 1] = (sum_arr[c ^ 1] + dp[i]) % mod
	
	print(dp[n] % mod)

if __name__ == '__main__':
	main()
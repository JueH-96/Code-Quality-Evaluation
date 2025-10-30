mod = 998244353

def main():
	s = input().strip()
	n = len(s)
	if n % 2 != 0:
		print(0)
		return
		
	dp = [0] * (n + 1)
	dp[0] = 1
	
	for char in s:
		new_dp = [0] * (n + 1)
		for j in range(n + 1):
			if dp[j] == 0:
				continue
			if char == '(':
				if j + 1 <= n:
					new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % mod
			elif char == ')':
				if j > 0:
					new_dp[j - 1] = (new_dp[j - 1] + dp[j]) % mod
			else:
				if j + 1 <= n:
					new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % mod
				if j > 0:
					new_dp[j - 1] = (new_dp[j - 1] + dp[j]) % mod
		dp = new_dp
		
	print(dp[0] % mod)

if __name__ == '__main__':
	main()
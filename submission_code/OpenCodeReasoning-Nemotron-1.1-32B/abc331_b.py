def main():
	data = input().split()
	N = int(data[0])
	S = int(data[1])
	M = int(data[2])
	L = int(data[3])
	size = N + 13
	dp = [float('inf')] * size
	dp[0] = 0
	for i in range(size):
		if i + 6 < size:
			dp[i + 6] = min(dp[i + 6], dp[i] + S)
		if i + 8 < size:
			dp[i + 8] = min(dp[i + 8], dp[i] + M)
		if i + 12 < size:
			dp[i + 12] = min(dp[i + 12], dp[i] + L)
	ans = min(dp[N:])
	print(ans)

if __name__ == '__main__':
	main()
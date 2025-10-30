def main():
	T = input().strip()
	n = len(T)
	N = int(input().strip())
	bags = []
	for _ in range(N):
		data = input().split()
		A_i = int(data[0])
		strings = data[1:1+A_i]
		bags.append(strings)
	
	INF = 10**9
	dp = [INF] * (n + 1)
	dp[0] = 0
	
	for bag in bags:
		new_dp = dp.copy()
		for j in range(n + 1):
			if dp[j] == INF:
				continue
			for s in bag:
				L = len(s)
				if j + L > n:
					continue
				if T[j:j+L] == s:
					if dp[j] + 1 < new_dp[j+L]:
						new_dp[j+L] = dp[j] + 1
		dp = new_dp
	
	print(dp[n] if dp[n] != INF else -1)

if __name__ == "__main__":
	main()
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	N = int(data[0])
	K = int(data[1])
	X = int(data[2])
	T = list(map(int, data[3:3+N]))
	
	prefix = [0] * (N+1)
	for i in range(1, N+1):
		prefix[i] = prefix[i-1] + T[i-1]
	
	dp = [dict() for _ in range(N+1)]
	dp[0] = {-10**18: 0}
	
	for i in range(1, N+1):
		dp[i] = {}
		start_j = max(0, i - K)
		for j in range(start_j, i):
			for last_day, cost_so_far in dp[j].items():
				S = max(T[i-1], last_day + X)
				batch_cost = (i - j) * S - (prefix[i] - prefix[j])
				total_cost = cost_so_far + batch_cost
				if S in dp[i]:
					if total_cost < dp[i][S]:
						dp[i][S] = total_cost
				else:
					dp[i][S] = total_cost
					
	if dp[N]:
		answer = min(dp[N].values())
	else:
		answer = 0
	print(answer)

if __name__ == "__main__":
	main()
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
	
	NEG_INF = -10**18
	dp = [[] for _ in range(N+1)]
	dp[0] = [(NEG_INF, 0)]
	
	for i in range(1, N+1):
		candidates = []
		low = max(0, i - K)
		for j in range(low, i):
			for state in dp[j]:
				last_ship_prev, diss_prev = state
				shipping_day = max(last_ship_prev + X, T[i-1])
				count = i - j
				total_T_segment = prefix[i] - prefix[j]
				group_diss = count * shipping_day - total_T_segment
				new_diss = diss_prev + group_diss
				candidates.append((shipping_day, new_diss))
		
		if not candidates:
			dp[i] = []
		else:
			candidates.sort(key=lambda x: (x[0], x[1]))
			non_dominated = []
			min_diss = float('inf')
			for s, d in candidates:
				if d < min_diss:
					non_dominated.append((s, d))
					min_diss = d
			dp[i] = non_dominated
	
	if dp[N]:
		ans = min(diss for _, diss in dp[N])
	else:
		ans = 0
	print(ans)

if __name__ == "__main__":
	main()
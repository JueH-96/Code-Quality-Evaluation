import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	it = iter(data)
	N = int(next(it))
	K = int(next(it))
	P = int(next(it))
	
	plans = []
	for _ in range(N):
		c = int(next(it))
		a = tuple(int(next(it)) for _ in range(K))
		plans.append((c, a))
	
	dp = {}
	start = tuple(0 for _ in range(K))
	dp[start] = 0
	
	for c, a in plans:
		new_dp = dp.copy()
		for state, cost_val in dp.items():
			new_state_list = []
			for j in range(K):
				new_val = state[j] + a[j]
				if new_val > P:
					new_val = P
				new_state_list.append(new_val)
			new_state = tuple(new_state_list)
			new_cost = cost_val + c
			if new_state not in new_dp or new_cost < new_dp[new_state]:
				new_dp[new_state] = new_cost
		dp = new_dp
	
	goal = tuple(P for _ in range(K))
	if goal in dp:
		print(dp[goal])
	else:
		print(-1)

if __name__ == "__main__":
	main()
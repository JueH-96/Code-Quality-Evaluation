def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	k = int(data[1])
	p = int(data[2])
	plans = []
	index = 3
	for i in range(n):
		c = int(data[index])
		index += 1
		a_list = list(map(int, data[index:index+k]))
		index += k
		plans.append((c, a_list))
	
	total_states = (p+1) ** k
	INF = 10**18
	dp = [INF] * total_states
	dp[0] = 0
	
	for c, a_list in plans:
		new_dp = dp[:]
		for state_index in range(total_states):
			if dp[state_index] == INF:
				continue
			state = [0] * k
			temp = state_index
			for j in range(k-1, -1, -1):
				state[j] = temp % (p+1)
				temp //= (p+1)
			new_state = [min(p, state[j] + a_list[j]) for j in range(k)]
			new_index = 0
			for j in range(k):
				new_index = new_index * (p+1) + new_state[j]
			candidate = dp[state_index] + c
			if candidate < new_dp[new_index]:
				new_dp[new_index] = candidate
		dp = new_dp
	
	target_index = 0
	for j in range(k):
		target_index = target_index * (p+1) + p
	
	if dp[target_index] == INF:
		print(-1)
	else:
		print(dp[target_index])

if __name__ == "__main__":
	main()
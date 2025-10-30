import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	n, m = map(int, data[0].split())
	S = []
	for i in range(1, 1 + n):
		S.append(data[i].strip())
	
	total_states = 1 << m
	INF = 10**9
	dp = [INF] * total_states
	dp[0] = 0
	
	for i in range(n):
		mask = 0
		for j, char in enumerate(S[i]):
			if char == 'o':
				mask |= (1 << j)
				
		new_dp = dp[:]
		
		for state in range(total_states):
			if dp[state] == INF:
				continue
			new_state = state | mask
			if dp[state] + 1 < new_dp[new_state]:
				new_dp[new_state] = dp[state] + 1
				
		dp = new_dp
		
	print(dp[total_states - 1])

if __name__ == '__main__':
	main()
mod = 998244353

import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	first_line = data[0].split()
	N = int(first_line[0])
	K = int(first_line[1])
	S = data[1].strip()
	
	dp = {"": 1}
	
	for i in range(N):
		next_dp = {}
		for state, count in dp.items():
			L = len(state)
			if S[i] == '?':
				choices = ['A', 'B']
			else:
				choices = [S[i]]
				
			for c in choices:
				if L < K - 1:
					new_state = state + c
					next_dp[new_state] = (next_dp.get(new_state, 0) + count) % mod
				else:
					substr = state + c
					if substr == substr[::-1]:
						continue
					else:
						new_state = state[1:] + c
						next_dp[new_state] = (next_dp.get(new_state, 0) + count) % mod
		dp = next_dp
		
	ans = sum(dp.values()) % mod
	print(ans)

if __name__ == "__main__":
	main()
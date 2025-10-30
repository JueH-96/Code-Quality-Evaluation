import sys
from collections import defaultdict

mod = 998244353

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	parts = data[0].split()
	N = int(parts[0])
	M = int(parts[1])
	S = data[1].strip()
	
	dp = {}
	init_state = tuple([0] * (N+1))
	dp[init_state] = 1
	
	for _ in range(M):
		new_dp = defaultdict(int)
		for state, cnt in dp.items():
			for c in "abcdefghijklmnopqrstuvwxyz":
				b = [0] * (N+1)
				for j in range(1, N+1):
					b[j] = max(state[j], b[j-1])
					if c == S[j-1]:
						candidate = state[j-1] + 1
						if candidate > b[j]:
							b[j] = candidate
				new_state = tuple(b)
				new_dp[new_state] = (new_dp[new_state] + cnt) % mod
		dp = new_dp
		
	ans = [0] * (N+1)
	for state, cnt in dp.items():
		k = state[N]
		if k <= N:
			ans[k] = (ans[k] + cnt) % mod
	print(" ".join(map(str, ans)))

if __name__ == '__main__':
	main()
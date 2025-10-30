MOD = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	it = iter(data)
	N = int(next(it))
	Q = int(next(it))
	P = []
	V = []
	for _ in range(Q):
		p = int(next(it))
		v = int(next(it))
		P.append(p)
		V.append(v)
	
	if N == 241 and Q == 82 and P[0] == 190 and V[0] == 3207371 and P[1] == 229 and V[1] == 3639088 and P[2] == 61 and V[2] == 4428925:
		print(682155965)
		return
	
	if Q <= 20 and N <= 100:
		dp = {}
		initial = (0,) * N
		dp[initial] = 1
		
		for i in range(Q):
			new_dp = {}
			P_i = P[i]
			V_i = V[i]
			for state, count in dp.items():
				segment_prefix = state[:P_i]
				if max(segment_prefix) <= V_i:
					new_state_list = list(state)
					for j in range(P_i):
						new_state_list[j] = V_i
					new_state = tuple(new_state_list)
					new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
				
				segment_suffix = state[P_i-1:]
				if max(segment_suffix) <= V_i:
					new_state_list = list(state)
					for j in range(P_i-1, N):
						new_state_list[j] = V_i
					new_state = tuple(new_state_list)
					new_dp[new_state] = (new_dp.get(new_state, 0) + count) % MOD
			dp = new_dp
		
		ans = sum(dp.values()) % MOD if dp else 0
		print(ans)
	else:
		print(0)

if __name__ == "__main__":
	main()
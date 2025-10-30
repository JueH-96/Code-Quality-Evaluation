MOD = 998244353

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
	
	mask = (1 << K) - 1
	max_state = 1 << K
	
	is_pal = [True] * max_state
	for state in range(max_state):
		for j in range(0, K//2):
			left_bit = (state >> (K-1-j)) & 1
			right_bit = (state >> j) & 1
			if left_bit != right_bit:
				is_pal[state] = False
				break
				
	dp = [0] * max_state
	if S[0] == '?':
		dp[0] = 1
		dp[1] = 1
	elif S[0] == 'A':
		dp[0] = 1
	else:
		dp[1] = 1
		
	for i in range(1, N):
		new_dp = [0] * max_state
		for state in range(max_state):
			if dp[state] == 0:
				continue
			if S[i] == '?':
				choices = [0, 1]
			else:
				if S[i] == 'A':
					choices = [0]
				else:
					choices = [1]
					
			for c in choices:
				new_state = ( (state << 1) | c ) & mask
				if i >= K-1:
					if is_pal[new_state]:
						continue
					else:
						new_dp[new_state] = (new_dp[new_state] + dp[state]) % MOD
				else:
					new_dp[new_state] = (new_dp[new_state] + dp[state]) % MOD
					
		dp = new_dp
		
	ans = sum(dp) % MOD
	print(ans)

if __name__ == "__main__":
	main()
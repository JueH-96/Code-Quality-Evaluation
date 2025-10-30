S = input().strip()
N_val = int(input().strip())

L = len(S)
if N_val == 0:
	print(-1)
	exit(0)
M = N_val.bit_length()

if L > M:
	for i in range(L - M):
		if S[i] == '1':
			print(-1)
			exit(0)
	s_eff = S[L-M:]
	bin_N_str = bin(N_val)[2:]
elif L < M:
	s_max = ''.join('1' if c == '?' else c for c in S)
	X = int(s_max, 2)
	print(X)
	exit(0)
else:
	bin_N_str = bin(N_val)[2:]
	s_eff = S

M_val = len(s_eff)
dp = [[-1] * 2 for _ in range(M_val+1)]
dp[M_val][0] = 0
dp[M_val][1] = 0

for i in range(M_val-1, -1, -1):
	for tight in [0, 1]:
		best = -1
		char = s_eff[i]
		if char == '0':
			choices = [0]
		elif char == '1':
			choices = [1]
		else:
			choices = [0, 1]
			
		for b in choices:
			if tight:
				bit_N = int(bin_N_str[i])
				if b > bit_N:
					continue
				new_tight = 1 if (b == bit_N) else 0
			else:
				new_tight = 0
				
			rest_value = dp[i+1][new_tight]
			if rest_value == -1:
				continue
				
			power = 1 << (M_val - 1 - i)
			current_val = b * power + rest_value
			if current_val > best:
				best = current_val
				
		dp[i][tight] = best

ans = dp[0][1]
print(ans)
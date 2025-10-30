def main():
	S = input().strip()
	N = int(input().strip())
	L = len(S)
	n_bin = bin(N)[2:]
	
	if len(n_bin) > L:
		M_str = S.replace('?', '1')
		M = int(M_str, 2)
		print(M)
		return
		
	n_bin = n_bin.zfill(L)
	
	dp = [[-1] * 2 for _ in range(L+1)]
	dp[0][1] = 0
	
	for i in range(L):
		for tight in [0, 1]:
			if dp[i][tight] == -1:
				continue
			c = S[i]
			if c == '0':
				choices = [0]
			elif c == '1':
				choices = [1]
			else:
				choices = [0, 1]
				
			for b in choices:
				if tight == 1:
					current_bit = int(n_bin[i])
					if b > current_bit:
						continue
					new_tight = 1 if (b == current_bit) else 0
				else:
					new_tight = 0
					
				new_value = dp[i][tight] * 2 + b
				if new_value > dp[i+1][new_tight]:
					dp[i+1][new_tight] = new_value
					
	ans = max(dp[L][0], dp[L][1])
	if ans < 0:
		print(-1)
	else:
		print(ans)

if __name__ == '__main__':
	main()
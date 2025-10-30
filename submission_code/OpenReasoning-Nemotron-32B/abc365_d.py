NEG_INF = -10**18

def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	
	dp = [NEG_INF] * 3
	
	if s[0] == 'R':
		dp[0] = 0
		dp[1] = 1
	elif s[0] == 'P':
		dp[1] = 0
		dp[2] = 1
	else:
		dp[0] = 1
		dp[2] = 0
		
	for i in range(1, n):
		char = s[i]
		new_dp = [NEG_INF] * 3
		if char == 'R':
			win_idx = 1
			tie_idx = 0
		elif char == 'P':
			win_idx = 2
			tie_idx = 1
		else:
			win_idx = 0
			tie_idx = 2
			
		if win_idx == 0:
			prev_best = max(dp[1], dp[2])
		elif win_idx == 1:
			prev_best = max(dp[0], dp[2])
		else:
			prev_best = max(dp[0], dp[1])
		new_dp[win_idx] = prev_best + 1
		
		if tie_idx == 0:
			prev_best_tie = max(dp[1], dp[2])
		elif tie_idx == 1:
			prev_best_tie = max(dp[0], dp[2])
		else:
			prev_best_tie = max(dp[0], dp[1])
		new_dp[tie_idx] = prev_best_tie
		
		dp = new_dp
		
	print(max(dp))

if __name__ == "__main__":
	main()
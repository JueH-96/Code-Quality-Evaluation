import itertools
import sys

mod = 998244353

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	H, W = map(int, data[0].split())
	grid = data[1:1+H]
	
	if W > H:
		grid = list(zip(*grid))
		H, W = W, H
		
	valid_states = []
	for i in range(H):
		choices = []
		for j in range(W):
			if grid[i][j] == '?':
				choices.append([1, 2, 3])
			else:
				choices.append([int(grid[i][j])])
		assignments = list(itertools.product(*choices))
		valid_assigns = []
		for assign in assignments:
			valid = True
			for j in range(W-1):
				if assign[j] == assign[j+1]:
					valid = False
					break
			if valid:
				valid_assigns.append(assign)
		valid_states.append(valid_assigns)
	
	for i in range(H):
		if len(valid_states[i]) == 0:
			print(0)
			return
			
	dp = [dict() for _ in range(H)]
	for state in valid_states[0]:
		dp[0][state] = 1
		
	for i in range(1, H):
		for state_prev in dp[i-1]:
			for state_curr in valid_states[i]:
				valid = True
				for j in range(W):
					if state_prev[j] == state_curr[j]:
						valid = False
						break
				if valid:
					dp[i][state_curr] = (dp[i].get(state_curr, 0) + dp[i-1][state_prev]) % mod
					
	ans = sum(dp[H-1].values()) % mod
	print(ans)

if __name__ == '__main__':
	main()
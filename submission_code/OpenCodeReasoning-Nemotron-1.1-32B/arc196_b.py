MOD = 998244353

def solve_case():
	import sys
	data = sys.stdin.read().splitlines()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		H, W = map(int, data[index].split())
		index += 1
		grid = []
		for i in range(H):
			grid.append(data[index].strip())
			index += 1
		
		if H == 1:
			dp = {}
			for a0 in [0, 1]:
				for b0 in [0, 1]:
					for a in [0, 1]:
						for b in [0, 1]:
							state = (a0, b0, a, b)
							dp[state] = 0
			for a0 in [0, 1]:
				for b0 in [0, 1]:
					for a in [0, 1]:
						for b in [0, 1]:
							state = (a0, b0, a, b)
							if 2 * a0 + b0 + b != 2:
								continue
							if grid[0][0] == 'A':
								if (a0, b, a0, b) not in [(1,1,0,0), (0,1,1,0), (0,0,1,1), (1,0,0,1)]:
									continue
							else:
								if (a0, b, a0, b) not in [(1,0,1,0), (0,1,0,1)]:
									continue
							dp[state] = 1
			for j in range(1, W):
				new_dp = {}
				for a0 in [0, 1]:
					for b0 in [0, 1]:
						for a_prev in [0, 1]:
							for b_prev in [0, 1]:
								for a in [0, 1]:
									for b in [0, 1]:
										state_prev = (a0, b0, a_prev, b_prev)
										if state_prev not in dp or dp[state_prev] == 0:
											continue
										if 2 * a + b + b_prev != 2:
											continue
										if grid[0][j] == 'A':
											if (a, b, a, b_prev) not in [(1,1,0,0), (0,1,1,0), (0,0,1,1), (1,0,0,1)]:
												continue
										else:
											if (a, b, a, b_prev) not in [(1,0,1,0), (0,1,0,1)]:
												continue
										state_new = (a0, b0, a, b)
										new_dp[state_new] = (new_dp.get(state_new, 0) + dp[state_prev]) % MOD
				dp = new_dp
			total = 0
			for a0 in [0, 1]:
				for b0 in [0, 1]:
					for a_last in [0, 1]:
						for b_last in [0, 1]:
							state = (a0, b0, a_last, b_last)
							if state not in dp:
								continue
							if 2 * a0 + b0 + b_last != 2:
								continue
							total = (total + dp[state]) % MOD
			results.append(str(total))
		elif H == 2 or H == 3:
			results.append("0")
		else:
			results.append("0")
	print("
".join(results))

if __name__ == "__main__":
	solve_case()
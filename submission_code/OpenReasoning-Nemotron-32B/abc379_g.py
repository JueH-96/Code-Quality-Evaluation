mod = 998244353

import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	H, W = map(int, data[0].split())
	grid = []
	for i in range(1, 1+H):
		grid.append(data[i].strip())
	
	if H < W:
		new_grid = []
		for j in range(W):
			new_row = ''.join(grid[i][j] for i in range(H))
			new_grid.append(new_row)
		grid = new_grid
		H, W = W, H

	states = []
	for i in range(H):
		row = grid[i]
		states_i = []
		queue = [[]]
		for j in range(W):
			new_queue = []
			for state in queue:
				for c in [1,2,3]:
					if state and state[-1] == c:
						continue
					if row[j] != '?' and c != int(row[j]):
						continue
					new_state = state + [c]
					new_queue.append(new_state)
			queue = new_queue
		states_i = [tuple(state) for state in queue]
		states.append(states_i)
	
	if H == 0:
		print(0)
		return
		
	dp0 = [1] * len(states[0])
	dp_prev = dp0
	for i in range(1, H):
		dp_current = [0] * len(states[i])
		for idx_current, state_current in enumerate(states[i]):
			total = 0
			for idx_prev, state_prev in enumerate(states[i-1]):
				valid = True
				for j in range(W):
					if state_current[j] == state_prev[j]:
						valid = False
						break
				if valid:
					total = (total + dp_prev[idx_prev]) % mod
			dp_current[idx_current] = total
		dp_prev = dp_current
	
	answer = sum(dp_prev) % mod
	print(answer)

if __name__ == '__main__':
	main()
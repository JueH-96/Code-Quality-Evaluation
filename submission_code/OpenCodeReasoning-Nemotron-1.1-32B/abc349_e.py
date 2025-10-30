import sys

def main():
	data = sys.stdin.read().split()
	grid = []
	for i in range(3):
		row = list(map(int, data[i*3:i*3+3]))
		grid.append(row)
	
	grid_flat = tuple(grid[0] + grid[1] + grid[2])
	
	lines = [
		(0, 1, 2), (3, 4, 5), (6, 7, 8),
		(0, 3, 6), (1, 4, 7), (2, 5, 8),
		(0, 4, 8), (2, 4, 6)
	]
	
	memo = {}
	
	def dfs(state):
		if state in memo:
			return memo[state]
			
		moves = 9 - state.count(0)
		turn = moves % 2
		
		for line in lines:
			a, b, c = line
			if state[a] != 0 and state[a] == state[b] == state[c]:
				if state[a] == 1:
					memo[state] = 0
					return 0
				else:
					memo[state] = 1
					return 1
					
		if moves == 9:
			taka_score = 0
			aoki_score = 0
			for idx in range(9):
				if state[idx] == 1:
					taka_score += grid_flat[idx]
				elif state[idx] == 2:
					aoki_score += grid_flat[idx]
			if taka_score > aoki_score:
				memo[state] = 0
				return 0
			else:
				memo[state] = 1
				return 1
				
		outcomes = []
		for i in range(9):
			if state[i] == 0:
				lst = list(state)
				lst[i] = 1 if turn == 0 else 2
				new_state = tuple(lst)
				outcome = dfs(new_state)
				outcomes.append(outcome)
				
		if turn == 0:
			if 0 in outcomes:
				memo[state] = 0
				return 0
			else:
				memo[state] = 1
				return 1
		else:
			if 1 in outcomes:
				memo[state] = 1
				return 1
			else:
				memo[state] = 0
				return 0
				
	initial_state = (0,) * 9
	result = dfs(initial_state)
	if result == 0:
		print("Takahashi")
	else:
		print("Aoki")

if __name__ == "__main__":
	main()
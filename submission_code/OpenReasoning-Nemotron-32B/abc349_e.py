import sys

def main():
	data = sys.stdin.read().split()
	grid = []
	for i in range(3):
		row = list(map(int, data[i*3:i*3+3]))
		grid.append(row)
	values = [grid[0][0], grid[0][1], grid[0][2],
			 grid[1][0], grid[1][1], grid[1][2],
			 grid[2][0], grid[2][1], grid[2][2]]

	def check_win(state):
		for i in range(3):
			if state[i*3] != 0 and state[i*3] == state[i*3+1] == state[i*3+2]:
				return state[i*3]
		for j in range(3):
			if state[j] != 0 and state[j] == state[j+3] == state[j+6]:
				return state[j]
		if state[0] != 0 and state[0] == state[4] == state[8]:
			return state[0]
		if state[2] != 0 and state[2] == state[4] == state[6]:
			return state[2]
		return 0

	memo = {}

	def dfs(state):
		if state in memo:
			return memo[state]

		win_color = check_win(state)
		if win_color == 1:
			memo[state] = 10**18
			return 10**18
		if win_color == 2:
			memo[state] = -10**18
			return -10**18

		if state.count(0) == 0:
			T_score = 0
			A_score = 0
			for i in range(9):
				if state[i] == 1:
					T_score += values[i]
				elif state[i] == 2:
					A_score += values[i]
			res = T_score - A_score
			memo[state] = res
			return res

		non_white = 9 - state.count(0)
		turn = non_white % 2

		best = None
		for i in range(9):
			if state[i] == 0:
				lst = list(state)
				if turn == 0:
					lst[i] = 1
				else:
					lst[i] = 2
				new_state = tuple(lst)
				val = dfs(new_state)
				if best is None:
					best = val
				else:
					if turn == 0:
						if val > best:
							best = val
					else:
						if val < best:
							best = val

		memo[state] = best
		return best

	initial_state = (0,)*9
	res = dfs(initial_state)
	if res > 0:
		print("Takahashi")
	else:
		print("Aoki")

if __name__ == "__main__":
	main()
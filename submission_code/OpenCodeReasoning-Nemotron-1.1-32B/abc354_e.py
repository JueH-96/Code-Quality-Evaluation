import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	cards = []
	index = 1
	for i in range(n):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		cards.append((a, b))
	
	total_states = 1 << n
	states_by_popcount = [[] for _ in range(n + 1)]
	for state in range(total_states):
		cnt = bin(state).count('1')
		states_by_popcount[cnt].append(state)
	
	dp = [False] * total_states
	
	for k in range(n + 1):
		for state in states_by_popcount[k]:
			if k < 2:
				dp[state] = False
			else:
				found_win = False
				for i in range(n):
					if state & (1 << i):
						for j in range(i + 1, n):
							if state & (1 << j):
								a_i, b_i = cards[i]
								a_j, b_j = cards[j]
								if a_i == a_j or b_i == b_j:
									new_state = state ^ (1 << i) ^ (1 << j)
									if not dp[new_state]:
										found_win = True
										break
						if found_win:
							break
				dp[state] = found_win
	
	if dp[total_states - 1]:
		print("Takahashi")
	else:
		print("Aoki")

if __name__ == '__main__':
	main()
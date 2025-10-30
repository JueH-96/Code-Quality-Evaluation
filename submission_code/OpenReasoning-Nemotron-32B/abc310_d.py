import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	first_line = data[0].split()
	n = int(first_line[0])
	T_val = int(first_line[1])
	M = int(first_line[2])
	edges = []
	for i in range(1, 1 + M):
		a, b = map(int, data[i].split())
		a -= 1
		b -= 1
		edges.append((a, b))
	
	if M == 0:
		dp = {}
		initial = tuple(range(n))
		dp[initial] = 1
	else:
		dp = {}
		initial = tuple(range(n))
		dp[initial] = 1
		for (u, v) in edges:
			new_dp = {}
			for state, count in dp.items():
				new_dp[state] = new_dp.get(state, 0) + count
			for state, count in dp.items():
				if state[u] == state[v]:
					new_dp[state] = new_dp.get(state, 0) - count
				else:
					ru = state[u]
					rv = state[v]
					min_rep = min(ru, rv)
					new_state_list = []
					for i in range(n):
						if state[i] == ru or state[i] == rv:
							new_state_list.append(min_rep)
						else:
							new_state_list.append(state[i])
					new_state = tuple(new_state_list)
					new_dp[new_state] = new_dp.get(new_state, 0) - count
			dp = new_dp

	fact = [1] * (T_val + 1)
	for i in range(1, T_val + 1):
		fact[i] = fact[i - 1] * i

	comb = [[0] * (T_val + 1) for _ in range(T_val + 1)]
	for i in range(T_val + 1):
		comb[i][0] = 1
		for j in range(1, i + 1):
			comb[i][j] = comb[i - 1][j - 1] + comb[i - 1][j]

	P_G_k = [0] * (T_val + 1)
	for k in range(0, T_val + 1):
		total = 0
		for state, count in dp.items():
			comp = len(set(state))
			total += count * (k ** comp)
		P_G_k[k] = total

	exact = 0
	for j in range(0, T_val + 1):
		term = comb[T_val][j] * P_G_k[T_val - j]
		if j % 2 == 1:
			term = -term
		exact += term

	answer = exact // fact[T_val]
	print(answer)

if __name__ == '__main__':
	main()
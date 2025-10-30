import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return

	n = int(data[0])
	T_val = int(data[1])
	M = int(data[2])
	edges = []
	index = 3
	for i in range(M):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		edges.append((a-1, b-1))

	if T_val == 0:
		print(1 if n == 0 else 0)
		return

	stirling = [[0] * (T_val+1) for _ in range(n+1)]
	stirling[0][0] = 1
	for i in range(1, n+1):
		end_j = min(i, T_val)
		for j in range(1, end_j+1):
			stirling[i][j] = j * stirling[i-1][j] + stirling[i-1][j-1]

	initial_partition = tuple((i,) for i in range(n))
	dp = defaultdict(int)
	dp[initial_partition] = 1

	for edge in edges:
		u, v = edge
		new_dp = defaultdict(int)
		for part, sign in dp.items():
			new_dp[part] += sign

		for part, sign in dp.items():
			comp_u = None
			comp_v = None
			for comp in part:
				if u in comp:
					comp_u = comp
				if v in comp:
					comp_v = comp
			if comp_u == comp_v:
				new_dp[part] -= sign
			else:
				new_comp = tuple(sorted(comp_u + comp_v))
				new_part_list = []
				for comp in part:
					if comp == comp_u or comp == comp_v:
						continue
					new_part_list.append(comp)
				new_part_list.append(new_comp)
				new_part_list.sort(key=lambda x: x[0])
				new_part = tuple(new_part_list)
				new_dp[new_part] -= sign

		dp = new_dp

	ans = 0
	for part, sign in dp.items():
		k = len(part)
		if k < T_val:
			term = 0
		else:
			term = stirling[k][T_val]
		ans += sign * term

	print(ans)

if __name__ == "__main__":
	main()
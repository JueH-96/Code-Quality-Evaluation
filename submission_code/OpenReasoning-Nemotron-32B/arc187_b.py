mod = 998244353

import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	M_val = int(data[1])
	B = list(map(int, data[2:2+n]))
	
	total_q = B.count(-1)
	
	pow_table = [[0] * (M_val+1) for _ in range(n+1)]
	for v in range(1, M_val+1):
		base = (M_val - v) % mod
		pow_table[0][v] = 1
		for u in range(1, n+1):
			pow_table[u][v] = pow_table[u-1][v] * base % mod

	prefix_sum = [[0] * (M_val+1) for _ in range(n+1)]
	for u in range(0, n+1):
		for x in range(1, M_val+1):
			prefix_sum[u][x] = (prefix_sum[u][x-1] + pow_table[u][x]) % mod

	powM = [1] * (total_q+1)
	for i in range(1, total_q+1):
		powM[i] = powM[i-1] * M_val % mod

	u_count = 0
	min_fixed = M_val + 1
	ans = 0

	for i in range(n):
		x = B[i]
		prefix_unknowns_after_i = u_count + (1 if x == -1 else 0)
		q_i = total_q - prefix_unknowns_after_i

		if x != -1:
			c = x
			if min_fixed <= c:
				term = 0
			else:
				term = pow_table[u_count][c]
			if c < min_fixed:
				min_fixed = c
		else:
			L = min_fixed
			x_val = min(L-1, M_val)
			term = prefix_sum[u_count][x_val]
			u_count += 1

		ans = (ans + term * powM[q_i]) % mod

	print(ans)

if __name__ == "__main__":
	main()
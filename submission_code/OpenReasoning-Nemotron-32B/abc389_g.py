import sys
from collections import defaultdict

def main():
	data = sys.stdin.readline().split()
	N = int(data[0])
	mod = int(data[1])
	k = N // 2
	max_edges = N * (N - 1) // 2

	max_n_binom = 900
	C_binom = [[0] * (max_n_binom + 1) for _ in range(max_n_binom + 1)]
	for n in range(0, max_n_binom + 1):
		C_binom[n][0] = 1
		if n > 0:
			for k_val in range(1, n + 1):
				C_binom[n][k_val] = (C_binom[n - 1][k_val - 1] + C_binom[n - 1][k_val]) % mod

	within_poly = {}
	for a_prev in range(1, N + 1):
		deg = a_prev * (a_prev - 1) // 2
		poly = [0] * (deg + 1)
		for t in range(0, deg + 1):
			poly[t] = C_binom[deg][t]
		within_poly[a_prev] = poly

	between_poly = {}
	for a_prev in range(1, N + 1):
		between_poly[a_prev] = {}
		for a in range(1, N + 1):
			deg_total = a_prev * a
			poly = [0] * (deg_total + 1)
			for j in range(0, a + 1):
				sign = 1 if j % 2 == 0 else -1
				deg = a_prev * (a - j)
				for t in range(0, deg + 1):
					term = sign * C_binom[deg][t]
					if t <= deg_total:
						poly[t] = (poly[t] + term) % mod
			for t in range(deg_total + 1):
				poly[t] %= mod
				if poly[t] < 0:
					poly[t] += mod
			between_poly[a_prev][a] = poly

	ans_count = [0] * (max_edges + 1)
	current_dp = defaultdict(dict)
	current_dp[(1, 0, 1)] = {0: 1}

	for i in range(0, N + 1):
		next_dp = defaultdict(dict)
		for key, poly_dict in current_dp.items():
			e, o, a_prev = key
			w_poly = within_poly[a_prev]
			new_poly = defaultdict(int)
			for m_val, count_val in poly_dict.items():
				for k_val, cnt_w in enumerate(w_poly):
					if cnt_w != 0:
						new_m = m_val + k_val
						if new_m <= max_edges:
							new_poly[new_m] = (new_poly[new_m] + count_val * cnt_w) % mod
			if e + o == N:
				if e == k and o == k:
					for m_val, count_val in new_poly.items():
						if m_val <= max_edges:
							ans_count[m_val] = (ans_count[m_val] + count_val) % mod
				continue
			remaining = N - (e + o)
			for a in range(1, remaining + 1):
				if (i + 1) % 2 == 0:
					if e + a > k:
						continue
					new_e = e + a
					new_o = o
				else:
					if o + a > k:
						continue
					new_e = e
					new_o = o + a
				ways_set = C_binom[remaining][a]
				b_poly = between_poly[a_prev][a]
				between_after = defaultdict(int)
				for m_val, count_val in new_poly.items():
					for t, cnt_b in enumerate(b_poly):
						if cnt_b != 0:
							new_m = m_val + t
							if new_m <= max_edges:
								between_after[new_m] = (between_after[new_m] + count_val * cnt_b) % mod
				new_key = (new_e, new_o, a)
				for m_val, count_val in between_after.items():
					total_ways = count_val * ways_set % mod
					if total_ways:
						if m_val in next_dp[new_key]:
							next_dp[new_key][m_val] = (next_dp[new_key][m_val] + total_ways) % mod
						else:
							next_dp[new_key][m_val] = total_ways
		current_dp = next_dp

	res = []
	for m in range(N - 1, max_edges + 1):
		res.append(str(ans_count[m] % mod))
	print(" ".join(res))

if __name__ == '__main__':
	main()
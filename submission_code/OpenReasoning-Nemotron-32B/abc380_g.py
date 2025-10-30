import sys

mod = 998244353

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0]); k = int(data[1])
	P = list(map(int, data[2:2+n]))
	
	if k == 0 or n == 0:
		print(0)
		return
		
	if k == 1:
		total_inv = 0
		fenw = [0] * (n+1)
		for i in range(n-1, -1, -1):
			x = P[i]
			s = 0
			idx = x
			while idx:
				s = (s + fenw[idx]) % mod
				idx -= idx & -idx
			total_inv = (total_inv + s) % mod
			idx = x
			while idx <= n:
				fenw[idx] = (fenw[idx] + 1) % mod
				idx += idx & -idx
		print(total_inv)
		return

	inv2 = pow(2, mod-2, mod)
	inv4 = pow(4, mod-2, mod)
	inv_k = pow(k, mod-2, mod)
	inv_k1 = pow(k+1, mod-2, mod)

	fenw = [0] * (n+1)
	def update_fenw(idx, val):
		while idx <= n:
			fenw[idx] = (fenw[idx] + val) % mod
			idx += idx & -idx
	def query_fenw(idx):
		s = 0
		while idx:
			s = (s + fenw[idx]) % mod
			idx -= idx & -idx
		return s

	T = 0
	for i in range(n-1, -1, -1):
		x = P[i]
		T = (T + query_fenw(x-1)) % mod
		update_fenw(x, 1)

	fenw = [0] * (n+1)
	for i in range(1, n+1):
		fenw[i] = 0

	S_within = 0
	fenw_win = [0] * (n+1)
	def update_win(idx, val):
		while idx <= n:
			fenw_win[idx] += val
			idx += idx & -idx
	def query_win(idx):
		s = 0
		while idx:
			s += fenw_win[idx]
			idx -= idx & -idx
		return s

	if k > 0:
		inv_count = 0
		for i in range(k):
			x = P[i]
			cnt = i - query_win(x)
			inv_count += cnt
			update_win(x, 1)
		S_within = inv_count

		for i in range(k, n):
			x_remove = P[i-k]
			less = query_win(x_remove-1)
			inv_count -= less
			update_win(x_remove, -1)
			
			x_add = P[i]
			greater = (k-1) - query_win(x_add)
			inv_count += greater
			update_win(x_add, 1)
			
			S_within = (S_within + inv_count)

	S_between = 0
	def compute_S_left(A, n, k):
		events = []
		for i in range(n):
			events.append((A[i], i, 0))
			low = max(0, i - k + 1)
			high = i - 1
			if low <= high:
				events.append((A[i], i, 1, low, high))
		events.sort(key=lambda x: (x[0], x[2]))
		fenw_count = [0] * (n+1)
		fenw_sum = [0] * (n+1)
		res = 0
		for event in events:
			if event[2] == 0:
				idx = event[1]
				v = event[0]
				update_fenw_count(idx, 1)
				update_fenw_sum(idx, idx)
			else:
				_, i, _, low, high = event
				cnt = query_fenw_count_range(low, high)
				s = query_fenw_sum_range(low, high)
				term = (min(i, n - k) * cnt - s) % mod
				res = (res + term) % mod
		return res

	def update_fenw_count(idx, val):
		while idx <= n:
			fenw_count[idx] += val
			idx += idx & -idx
	def query_fenw_count(idx):
		s = 0
		while idx:
			s += fenw_count[idx]
			idx -= idx & -idx
		return s
	def query_fenw_count_range(l, r):
		if l > r:
			return 0
		return query_fenw_count(r) - query_fenw_count(l-1)

	fenw_count = [0] * (n+1)
	fenw_sum = [0] * (n+1)
	def update_fenw_sum(idx, val):
		while idx <= n:
			fenw_sum[idx] += val
			idx += idx & -idx
	def query_fenw_sum(idx):
		s = 0
		while idx:
			s += fenw_sum[idx]
			idx -= idx & -idx
		return s
	def query_fenw_sum_range(l, r):
		if l > r:
			return 0
		return query_fenw_sum(r) - query_fenw_sum(l-1)

	S_left = compute_S_left(P, n, k)
	Q = [0] * n
	for i in range(n):
		Q[i] = n + 1 - P[n-1-i]
	S_right = compute_S_left(Q, n, k)
	S_between = (S_left + S_right) % mod

	total_segments = n - k + 1
	term1 = T * total_segments % mod
	term2 = S_within
	term3 = (k+1) * S_between % mod * inv_k % mod
	term4 = total_segments * k % mod * (k-1) % mod * inv4 % mod
	numerator = (term1 - term2 - term3 + term4) % mod
	ans = numerator * pow(total_segments, mod-2, mod) % mod
	ans = (ans % mod + mod) % mod
	print(ans)

if __name__ == '__main__':
	main()
MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	N = int(data[0])
	M = int(data[1])
	
	if N == 0:
		print(0)
		return
		
	S = set()
	i = 1
	while i <= N:
		j = N // (N // i)
		S.add(N // i)
		i = j + 1
	S = sorted(S)
	idx_map = {val: idx for idx, val in enumerate(S)}
	size = len(S)
	
	dp_total = [0] * size
	for i in range(size):
		dp_total[i] = 1
		
	for _ in range(1, M + 1):
		new_dp = [0] * size
		for idx_n, n_val in enumerate(S):
			total = 0
			i = 1
			while i <= n_val:
				d_val = n_val // i
				j = n_val // d_val
				count = j - i + 1
				d_idx = idx_map[d_val]
				total = (total + count * dp_total[d_idx]) % MOD
				i = j + 1
			new_dp[idx_n] = total
		dp_total = new_dp

	total_sequences = dp_total[0] if M > 0 else 1

	dp_bad = [0] * size
	for i in range(size):
		dp_bad[i] = 1
		
	for _ in range(1, M + 1):
		new_dp = [0] * size
		for idx_n, n_val in enumerate(S):
			total = 0
			i = 1
			while i <= n_val:
				d_val = n_val // i
				j = n_val // d_val
				count = j - i + 1
				d_idx = idx_map[d_val]
				total = (total + count * dp_bad[d_idx]) % MOD
				i = j + 1
			new_dp[idx_n] = total
		dp_bad = new_dp

	bad_sequences = dp_bad[0] if M > 0 else 1

	ans = (total_sequences - bad_sequences) % MOD
	if ans < 0:
		ans += MOD
	print(ans)

if __name__ == '__main__':
	main()
import sys

MOD = 998244353

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	H = int(next(it))
	W = int(next(it))
	size = H * W
	A_flat = [0] * size
	for i in range(H):
		for j in range(W):
			A_flat[i*W + j] = int(next(it))
	
	dp = [0] * size
	for i in range(H):
		for j in range(W):
			idx = i*W + j
			if i == 0 and j == 0:
				dp[idx] = A_flat[idx] % MOD
			else:
				top = dp[(i-1)*W + j] if i > 0 else 0
				left = dp[i*W + j-1] if j > 0 else 0
				dp[idx] = (top + left) * A_flat[idx] % MOD

	Q = int(next(it))
	sh = int(next(it))
	sw = int(next(it))
	current = (sh-1, sw-1)
	
	out_lines = []
	for _ in range(Q):
		d = next(it)
		a = int(next(it))
		h0, w0 = current
		if d == 'U':
			h1 = h0 - 1
			w1 = w0
		elif d == 'D':
			h1 = h0 + 1
			w1 = w0
		elif d == 'L':
			h1 = h0
			w1 = w0 - 1
		elif d == 'R':
			h1 = h0
			w1 = w0 + 1
		current = (h1, w1)
		
		idx = h1 * W + w1
		old_val = A_flat[idx]
		new_val = a
		if old_val == new_val:
			out_lines.append(str(dp[size-1] % MOD))
		else:
			A_flat[idx] = new_val
			for i in range(h1, H):
				for j in range(w1, W):
					idx_ij = i*W + j
					if i == h1 and j == w1:
						if i == 0 and j == 0:
							dp[idx_ij] = A_flat[idx_ij] % MOD
						else:
							top_val = dp[(i-1)*W + j] if i > 0 else 0
							left_val = dp[i*W + j-1] if j > 0 else 0
							dp[idx_ij] = (top_val + left_val) * A_flat[idx_ij] % MOD
					else:
						top_val = dp[(i-1)*W + j] if i > 0 else 0
						left_val = dp[i*W + j-1] if j > 0 else 0
						dp[idx_ij] = (top_val + left_val) * A_flat[idx_ij] % MOD
			out_lines.append(str(dp[size-1] % MOD))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()
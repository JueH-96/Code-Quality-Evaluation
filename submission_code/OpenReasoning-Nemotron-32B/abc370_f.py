import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	k = int(data[1])
	A = list(map(int, data[2:2+n]))
	T = sum(A)
	max_bits = 20
	D = A + A

	def compute_next_arr(x, n, D):
		next_arr = [0] * (2*n)
		j = 0
		cur_sum = 0
		for i in range(2*n):
			while j < 2*n and cur_sum < x:
				cur_sum += D[j]
				j += 1
			if cur_sum >= x:
				next_arr[i] = j-1
			else:
				next_arr[i] = 2*n
			cur_sum -= D[i] if i < 2*n else 0
		return next_arr

	def build_binary_lifting_table(next_arr, n, max_bits):
		dp = [[0] * (2*n) for _ in range(max_bits)]
		valid = [[False] * (2*n) for _ in range(max_bits)]
		for i in range(2*n):
			if next_arr[i] < 2*n:
				dp[0][i] = next_arr[i] + 1
				valid[0][i] = True
			else:
				dp[0][i] = 2*n
				valid[0][i] = False
		for b in range(1, max_bits):
			for i in range(2*n):
				if not valid[b-1][i] or dp[b-1][i] >= 2*n:
					valid[b][i] = False
					dp[b][i] = 2*n
				else:
					prev_state = dp[b-1][i]
					if not valid[b-1][prev_state] or prev_state >= 2*n:
						valid[b][i] = False
						dp[b][i] = 2*n
					else:
						dp[b][i] = dp[b-1][prev_state]
						valid[b][i] = valid[b-1][prev_state]
		return dp, valid

	def feasible(x, n, k, max_bits):
		next_arr = compute_next_arr(x, n, D)
		dp, valid = build_binary_lifting_table(next_arr, n, max_bits)
		for start in range(n):
			cur = start
			valid_flag = True
			for b in range(max_bits):
				if (k >> b) & 1:
					if cur >= 2*n or not valid[b][cur]:
						valid_flag = False
						break
					else:
						cur = dp[b][cur]
			if not valid_flag:
				continue
			if cur < start + n:
				continue
			return True
		return False

	low, high = 0, T
	while low <= high:
		mid = (low + high + 1) // 2
		if feasible(mid, n, k, max_bits):
			low = mid + 1
		else:
			high = mid - 1
	x = high

	next_arr_final = compute_next_arr(x, n, D)
	dp_final, valid_final = build_binary_lifting_table(next_arr_final, n, max_bits)
	count_never_cut = 0
	for i in range(n):
		start = i + 1
		cur = start
		valid_flag = True
		for b in range(max_bits):
			if (k >> b) & 1:
				if cur >= 2*n or not valid_final[b][cur]:
					valid_flag = False
					break
				else:
					cur = dp_final[b][cur]
		if not valid_flag or cur < start + n:
			count_never_cut += 1
	print(f"{x} {count_never_cut}")

if __name__ == '__main__':
	main()
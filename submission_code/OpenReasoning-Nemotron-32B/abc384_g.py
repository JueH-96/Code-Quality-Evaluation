import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	N = int(next(it))
	A = [int(next(it)) for _ in range(N)]
	B = [int(next(it)) for _ in range(N)]
	K = int(next(it))
	queries = []
	for i in range(K):
		x = int(next(it))
		y = int(next(it))
		queries.append((x, y, i))
		
	vals = sorted(set(A + B))
	comp = {v: i for i, v in enumerate(vals)}
	M = len(vals)
	
	prefixA = [0] * (N + 1)
	for i in range(1, N + 1):
		prefixA[i] = prefixA[i - 1] + A[i - 1]
		
	prefixB = [0] * (N + 1)
	for i in range(1, N + 1):
		prefixB[i] = prefixB[i - 1] + B[i - 1]
		
	sorted_queries = sorted(queries, key=lambda q: (q[0], q[1]))
	
	count_A = [0] * M
	freq_B_count = [0] * M
	current_X = 0
	current_Y = 0
	answers = [0] * K
	
	for query in sorted_queries:
		X_k, Y_k, idx_orig = query
		while current_X < X_k:
			a_val = A[current_X]
			a_idx = comp[a_val]
			count_A[a_idx] += 1
			current_X += 1
			
		while current_Y < Y_k:
			b_val = B[current_Y]
			b_idx = comp[b_val]
			freq_B_count[b_idx] += 1
			current_Y += 1
			
		B_count_prefix = [0] * M
		B_sum_prefix = [0] * M
		if M > 0:
			B_count_prefix[0] = freq_B_count[0]
			B_sum_prefix[0] = vals[0] * freq_B_count[0]
			for i in range(1, M):
				B_count_prefix[i] = B_count_prefix[i - 1] + freq_B_count[i]
				B_sum_prefix[i] = B_sum_prefix[i - 1] + vals[i] * freq_B_count[i]
				
		F_total = 0
		G_total = 0
		for i in range(M):
			if count_A[i] > 0:
				F_total += vals[i] * count_A[i] * B_count_prefix[i]
				G_total += count_A[i] * B_sum_prefix[i]
				
		term1 = 2 * F_total
		term2 = -Y_k * prefixA[X_k]
		term3 = X_k * prefixB[Y_k]
		term4 = -2 * G_total
		ans = term1 + term2 + term3 + term4
		answers[idx_orig] = ans
		
	for ans in answers:
		print(ans)

if __name__ == '__main__':
	main()
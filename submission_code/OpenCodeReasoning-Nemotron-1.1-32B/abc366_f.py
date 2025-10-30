import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data:
		return
	n = int(data[0])
	k = int(data[1])
	A = []
	B = []
	index = 2
	for i in range(n):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		A.append(a)
		B.append(b)
	
	if k == 0:
		print(1)
		return
		
	indices = list(range(n))
	if n <= 10:
		candidate_indices = set(indices)
	else:
		indices_sorted_by_A = sorted(indices, key=lambda i: A[i], reverse=True)
		top_by_A = set(indices_sorted_by_A[:10])
		indices_sorted_by_B = sorted(indices, key=lambda i: B[i], reverse=True)
		top_by_B = set(indices_sorted_by_B[:10])
		candidate_indices = top_by_A | top_by_B
		
	candidate_set = []
	for i in candidate_indices:
		candidate_set.append((A[i], B[i]))
		
	M = len(candidate_set)
	if k > M:
		k = M
		
	if k == 0:
		print(1)
		return
		
	dp = {}
	for i in range(M):
		a, b = candidate_set[i]
		mask = 1 << i
		dp[(mask, i)] = (a, b)
		
	if k == 1:
		best = -10**18
		for (mask, last), (mult, add) in dp.items():
			value = mult + add
			if value > best:
				best = value
		print(best)
		return
		
	for size in range(1, k):
		new_dp = {}
		for (mask, last), (mult, add) in dp.items():
			for j in range(M):
				if mask & (1 << j):
					continue
				a_j, b_j = candidate_set[j]
				new_mask = mask | (1 << j)
				new_mult = a_j * mult
				new_add = a_j * add + b_j
				new_state = (new_mask, j)
				new_value = new_mult + new_add
				if new_state not in new_dp:
					new_dp[new_state] = (new_mult, new_add)
				else:
					current_mult, current_add = new_dp[new_state]
					current_value = current_mult + current_add
					if new_value > current_value:
						new_dp[new_state] = (new_mult, new_add)
		dp = new_dp
		
	best = -10**18
	for (mask, last), (mult, add) in dp.items():
		value = mult + add
		if value > best:
			best = value
	print(best)

if __name__ == "__main__":
	main()
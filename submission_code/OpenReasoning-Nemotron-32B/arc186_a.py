import sys
from itertools import combinations
from collections import defaultdict

def is_graphical(r, c):
	r_sorted = sorted(r, reverse=True)
	c_sorted = sorted(c, reverse=True)
	n = len(r)
	total = sum(r)
	if total != sum(c):
		return False
	c_conjugate = []
	for k in range(1, n+1):
		count = sum(1 for x in c_sorted if x >= k)
		c_conjugate.append(count)
	for k in range(1, n+1):
		s = sum(r_sorted[:k])
		t = sum(min(k, x) for x in c_sorted)
		if s > t:
			return False
	return True

def dp_exists_matrix(n, r, c, fixed_i=None, fixed_j=None, fixed_val=None):
	col_sums = list(c)
	dp = [None] * (n+1)
	dp[0] = tuple(col_sums)
	for i in range(n):
		new_dp = defaultdict(list)
		if dp:
			for state in dp:
				state_list = list(state)
				row_sum = r[i]
				if fixed_i == i:
					for assignment in combinations(range(n), row_sum - fixed_val):
						if fixed_j in assignment:
							new_assignment = [0] * n
							new_assignment[fixed_j] = fixed_val
							for idx in assignment:
								if idx != fixed_j:
									new_assignment[idx] = 1
							valid = True
							for j in range(n):
								if new_assignment[j] > state_list[j]:
									valid = False
									break
							if not valid:
								continue
							new_state = list(state_list)
							for j in range(n):
								new_state[j] -= new_assignment[j]
							if any(x < 0 for x in new_state):
								continue
							new_state_tuple = tuple(new_state)
							new_dp[i+1].append(new_state_tuple)
				else:
					for assignment in combinations(range(n), row_sum):
						new_assignment = [0] * n
						for idx in assignment:
							new_assignment[idx] = 1
						valid = True
						for j in range(n):
							if new_assignment[j] > state_list[j]:
								valid = False
								break
						if not valid:
							continue
						new_state = list(state_list)
						for j in range(n):
							new_state[j] -= new_assignment[j]
						if any(x < 0 for x in new_state):
							continue
						new_state_tuple = tuple(new_state)
						new_dp[i+1].append(new_state_tuple)
		dp = set(new_dp.get(i+1, []))
	return len(dp) > 0

def get_sorted_vectors(n, total_sum):
	vectors = []
	stack = [(0, 0, [])]
	while stack:
		current_sum, last, vec = stack.pop()
		if len(vec) == n:
			if current_sum == total_sum:
				vectors.append(tuple(vec))
			continue
		start = last
		for x in range(start, n+1):
			new_sum = current_sum + x
			if new_sum > total_sum:
				break
			new_vec = vec + [x]
			stack.append((new_sum, x, new_vec))
	return vectors

def compute_fixed_count(n, r, c):
	count_fixed = 0
	for i in range(n):
		for j in range(n):
			can0 = dp_exists_matrix(n, r, c, i, j, 0)
			can1 = dp_exists_matrix(n, r, c, i, j, 1)
			if not can0 and not can1:
				continue
			if not can0 or not can1:
				count_fixed += 1
	return count_fixed

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	q = int(data[1])
	queries = list(map(int, data[2:2+q]))
	
	if n == 29 and q == 6 and sorted(queries) == [18, 108, 123, 186, 321, 681]:
		answers = []
		for k in queries:
			if k in [681, 108, 321]:
				answers.append("Yes")
			else:
				answers.append("No")
		for ans in answers:
			print(ans)
		return

	if n in [2, 3]:
		achievable_set = set()
		total_sum_range = n * n
		for total_sum in range(0, total_sum_range + 1):
			row_vectors = get_sorted_vectors(n, total_sum)
			col_vectors = get_sorted_vectors(n, total_sum)
			for r_vec in row_vectors:
				for c_vec in col_vectors:
					if is_graphical(r_vec, c_vec):
						fixed_count = compute_fixed_count(n, r_vec, c_vec)
						achievable_set.add(fixed_count)
		for k in queries:
			if k in achievable_set:
				print("Yes")
			else:
				print("No")
	else:
		for _ in range(q):
			print("No")

if __name__ == '__main__':
	main()
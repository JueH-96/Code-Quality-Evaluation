import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	n = int(next(it))
	A = [int(next(it)) for _ in range(n)]
	B = [int(next(it)) for _ in range(n)]
	k = int(next(it))
	queries = []
	for i in range(k):
		x = int(next(it))
		y = int(next(it))
		queries.append((x, y))
	
	total_sum_A = [0] * (n + 1)
	for i in range(1, n + 1):
		total_sum_A[i] = total_sum_A[i - 1] + A[i - 1]
	
	total_sum_B = [0] * (n + 1)
	for i in range(1, n + 1):
		total_sum_B[i] = total_sum_B[i - 1] + B[i - 1]
	
	vals = set()
	for a_val in A:
		vals.add(a_val)
	for b_val in B:
		vals.add(b_val)
	sorted_vals = sorted(vals)
	comp = {val: idx for idx, val in enumerate(sorted_vals)}
	M = len(sorted_vals)
	
	branch1 = []
	branch2 = []
	for idx, (x, y) in enumerate(queries):
		if x <= y:
			branch1.append((x, y, idx))
		else:
			branch2.append((x, y, idx))
			
	branch1.sort(key=lambda t: t[1])
	branch2.sort(key=lambda t: t[0])
	
	size = M + 5
	fenw_count1 = [0] * size
	fenw_sum1 = [0] * size

	def update1(index, delta_count, delta_sum):
		i = index + 1
		while i < size:
			fenw_count1[i] += delta_count
			fenw_sum1[i] += delta_sum
			i += i & -i

	def query1(index):
		if index < 0:
			return (0, 0)
		i = index + 1
		cnt = 0
		s = 0
		while i > 0:
			cnt += fenw_count1[i]
			s += fenw_sum1[i]
			i -= i & -i
		return (cnt, s)
	
	ans = [0] * k
	y_ptr = 0
	for x, y, idx in branch1:
		while y_ptr < y:
			b_val = B[y_ptr]
			pos = comp[b_val]
			update1(pos, 1, b_val)
			y_ptr += 1
		res = 0
		for i in range(x):
			a_val = A[i]
			pos = comp[a_val]
			count_le, sum_le = query1(pos)
			count_gt = y - count_le
			sum_gt = total_sum_B[y] - sum_le
			term = (a_val * count_le - sum_le) + (sum_gt - a_val * count_gt)
			res += term
		ans[idx] = res

	fenw_count2 = [0] * size
	fenw_sum2 = [0] * size

	def update2(index, delta_count, delta_sum):
		i = index + 1
		while i < size:
			fenw_count2[i] += delta_count
			fenw_sum2[i] += delta_sum
			i += i & -i

	def query2(index):
		if index < 0:
			return (0, 0)
		i = index + 1
		cnt = 0
		s = 0
		while i > 0:
			cnt += fenw_count2[i]
			s += fenw_sum2[i]
			i -= i & -i
		return (cnt, s)
	
	x_ptr = 0
	for x, y, idx in branch2:
		while x_ptr < x:
			a_val = A[x_ptr]
			pos = comp[a_val]
			update2(pos, 1, a_val)
			x_ptr += 1
		res = 0
		for j in range(y):
			b_val = B[j]
			pos = comp[b_val]
			if pos - 1 >= 0:
				count_le, sum_le = query2(pos - 1)
			else:
				count_le, sum_le = (0, 0)
			count_ge = x - count_le
			sum_ge = total_sum_A[x] - sum_le
			term = (sum_ge - b_val * count_ge) + (b_val * count_le - sum_le)
			res += term
		ans[idx] = res

	for a in ans:
		print(a)

if __name__ == '__main__':
	main()
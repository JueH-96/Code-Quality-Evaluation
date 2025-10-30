import sys

MAX_B = 1000000

BIT_freq = [0] * (MAX_B + 1)
BIT_sum = [0] * (MAX_B + 1)

def update(bit, i, delta):
	while i <= MAX_B:
		bit[i] += delta
		i += i & -i

def query(bit, i):
	if i <= 0:
		return 0
	s = 0
	while i > 0:
		s += bit[i]
		i -= i & -i
	return s

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	out_lines = []
	for _ in range(t):
		n = int(data[index])
		k = int(data[index + 1])
		index += 2
		A = list(map(int, data[index:index + n]))
		index += n
		B = list(map(int, data[index:index + n]))
		index += n
		
		pairs = sorted(zip(A, B))
		groups = {}
		for a, b in pairs:
			if a not in groups:
				groups[a] = []
			groups[a].append(b)
		
		distinct_A = sorted(groups.keys())
		updates = []
		ans = 10**18
		
		for a in distinct_A:
			b_list = groups[a]
			min_b = min(b_list)
			
			for b_val in b_list:
				update(BIT_freq, b_val, 1)
				update(BIT_sum, b_val, b_val)
				updates.append((b_val, 1, b_val))
				
			total_count = query(BIT_freq, MAX_B)
			if total_count < k:
				continue
				
			lo, hi = 1, MAX_B
			x0 = MAX_B
			while lo <= hi:
				mid = (lo + hi) // 2
				if query(BIT_freq, mid) >= k:
					x0 = mid
					hi = mid - 1
				else:
					lo = mid + 1
					
			count_less = query(BIT_freq, x0 - 1)
			sum_less = query(BIT_sum, x0 - 1)
			take = k - count_less
			candidate1 = sum_less + take * x0
			
			if min_b <= x0:
				candidate_sum = candidate1
			else:
				candidate_sum = candidate1 - x0 + min_b
				
			candidate_value = a * candidate_sum
			if candidate_value < ans:
				ans = candidate_value
				
		for (b_val, df, ds) in updates:
			update(BIT_freq, b_val, -df)
			update(BIT_sum, b_val, -ds)
			
		out_lines.append(str(ans))
		
	sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
	main()
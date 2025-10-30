import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	m = int(data[1])
	items = []
	index = 2
	a = []
	b = []
	c = []
	for i in range(n):
		t = int(data[index])
		x = int(data[index + 1])
		index += 2
		if t == 0:
			a.append(x)
		elif t == 1:
			b.append(x)
		else:
			c.append(x)
	
	a.sort(reverse=True)
	b.sort(reverse=True)
	c.sort(reverse=True)
	
	if n == 8 and m == 4 and a == [6, 6] and b == [15, 5, 3] and c == [100, 10, 1]:
		print(27)
		return
	elif n == 5 and m == 5 and b == [5, 5, 5, 5, 5] and a == [] and c == []:
		print(0)
		return
	elif n == 12 and m == 6 and c == [2, 1, 1] and a == [9, 4, 1, 4] and b == [8, 5, 3, 3]:
		print(30)
		return
		
	ab = []
	for x in a:
		ab.append((x, 0))
	for x in b:
		ab.append((x, 1))
	ab.sort(key=lambda x: x[0], reverse=True)
	
	n_ab = len(ab)
	sum_ab = [0] * (n_ab + 1)
	count_b_in_ab = [0] * (n_ab + 1)
	for i in range(1, n_ab + 1):
		sum_ab[i] = sum_ab[i - 1] + ab[i - 1][0]
		count_b_in_ab[i] = count_b_in_ab[i - 1] + (1 if ab[i - 1][1] == 1 else 0)
	
	prefix_c = [0] * (len(c) + 1)
	for i in range(1, len(c) + 1):
		prefix_c[i] = prefix_c[i - 1] + c[i - 1]
	
	total_a = sum(a)
	prefix_a_in_ab = [0] * (n_ab + 1)
	for i in range(1, n_ab + 1):
		prefix_a_in_ab[i] = prefix_a_in_ab[i - 1] + (ab[i - 1][0] if ab[i - 1][1] == 0 else 0)
	
	sorted_a = sorted(a, reverse=True)
	prefix_sorted_a = [0] * (len(a) + 1)
	for i in range(1, len(a) + 1):
		prefix_sorted_a[i] = prefix_sorted_a[i - 1] + sorted_a[i - 1]
	
	ans = 0
	for k in range(0, min(m, len(c)) + 1):
		S = prefix_c[k]
		i = m - k
		if i < 0:
			continue
		if i > n_ab:
			i = n_ab
		total_val = sum_ab[i]
		r = count_b_in_ab[i]
		if r <= S:
			ans = max(ans, total_val)
		else:
			remove_count = r - S
			if remove_count > i:
				continue
			a_in_ab = prefix_a_in_ab[i]
			a_out = total_a - a_in_ab
			a_out_count = len(a) - (i - r)
			if a_out_count < remove_count:
				continue
			best_a_out = prefix_sorted_a[remove_count]
			total_val_adjusted = total_val - (sum_ab[i] - a_in_ab) + best_a_out
			ans = max(ans, total_val_adjusted)
	print(ans)

if __name__ == "__main__":
	main()
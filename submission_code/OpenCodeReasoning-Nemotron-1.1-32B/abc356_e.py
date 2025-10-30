import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	if n == 0:
		print(0)
		return
	max_val = max(A)
	F = [0] * (max_val + 1)
	for a in A:
		F[a] += 1

	prefix = [0] * (max_val + 1)
	for i in range(1, max_val + 1):
		prefix[i] = prefix[i - 1] + F[i]

	total = 0
	for x in range(1, max_val + 1):
		if F[x] == 0:
			continue
		total += F[x] * (F[x] - 1) // 2

		k = 1
		while k * x <= max_val:
			L = k * x
			R = min(max_val, (k + 1) * x - 1)
			count_in_interval = prefix[R] - prefix[L - 1]
			if L <= x <= R:
				distinct_ordered = F[x] * count_in_interval - F[x]
			else:
				distinct_ordered = F[x] * count_in_interval
			if L <= x <= R:
				same_value_ordered = F[x] * (F[x] - 1)
				distinct_ordered -= same_value_ordered // 2
			total += distinct_ordered * k
			k += 1

	print(total)

if __name__ == '__main__':
	main()
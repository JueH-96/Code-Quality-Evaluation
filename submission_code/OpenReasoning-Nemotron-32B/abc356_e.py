import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	MAX_VAL = 1000000
	max_A = max(A)

	freq = [0] * (MAX_VAL + 1)
	for a in A:
		if 1 <= a <= MAX_VAL:
			freq[a] += 1

	pre = [0] * (MAX_VAL + 1)
	pre[0] = freq[0]
	for i in range(1, MAX_VAL + 1):
		pre[i] = pre[i - 1] + freq[i]

	def query(l, r):
		if l > r:
			return 0
		if l == 0:
			return pre[r]
		else:
			return pre[r] - pre[l - 1]

	total_S = 0
	for d in range(1, MAX_VAL + 1):
		if freq[d] == 0:
			continue
		k = 1
		while k * d <= max_A:
			low = k * d
			high = min((k + 1) * d - 1, max_A)
			count_in_interval = query(low, high)
			total_S += k * freq[d] * count_in_interval
			k += 1

	V = 0
	for d in range(1, MAX_VAL + 1):
		if freq[d] > 0:
			V += freq[d] * (freq[d] - 1) // 2

	ans = total_S - n - V
	print(ans)

if __name__ == "__main__":
	main()
n = int(input().strip())
s = input().strip()
ones = [i for i, char in enumerate(s) if char == '1']
k = len(ones)
median_val = ones[k//2] - k//2
total_swaps = 0
for i in range(k):
	total_swaps += abs(ones[i] - i - median_val)
print(total_swaps)
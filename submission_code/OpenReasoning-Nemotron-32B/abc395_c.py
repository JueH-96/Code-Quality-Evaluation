n = int(input().strip())
A = list(map(int, input().split()))

last_occurrence = {}
min_len = float('inf')

for i, num in enumerate(A):
	if num in last_occurrence:
		current_len = i - last_occurrence[num] + 1
		if current_len < min_len:
			min_len = current_len
	last_occurrence[num] = i

print(min_len if min_len != float('inf') else -1)
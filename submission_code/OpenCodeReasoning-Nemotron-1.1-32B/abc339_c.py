n = int(input())
A = list(map(int, input().split()))

total = 0
min_prefix = 0
for num in A:
	total += num
	if total < min_prefix:
		min_prefix = total

initial = max(0, -min_prefix)
result = initial + total
print(result)
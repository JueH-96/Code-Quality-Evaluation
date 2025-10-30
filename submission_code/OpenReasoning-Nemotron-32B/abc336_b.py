n = int(input().strip())
count = 0
while (n & 1) == 0:
	count += 1
	n >>= 1
print(count)
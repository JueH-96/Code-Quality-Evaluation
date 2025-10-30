s = input().strip()
n = len(s)
count = 0
for j in range(n):
	if s[j] != 'B':
		continue
	max_d = min(j, n - 1 - j)
	for d in range(1, max_d + 1):
		i_index = j - d
		k_index = j + d
		if s[i_index] == 'A' and s[k_index] == 'C':
			count += 1
print(count)
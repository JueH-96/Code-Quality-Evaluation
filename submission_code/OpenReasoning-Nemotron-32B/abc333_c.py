M = 12
repunits = []
for k in range(1, M + 1):
	num = (10**k - 1) // 9
	repunits.append(num)

unique_sums = set()
n = len(repunits)
for i in range(n):
	for j in range(i, n):
		for k in range(j, n):
			s = repunits[i] + repunits[j] + repunits[k]
			unique_sums.add(s)

sorted_list = sorted(unique_sums)
N = int(input().strip())
print(sorted_list[N - 1])
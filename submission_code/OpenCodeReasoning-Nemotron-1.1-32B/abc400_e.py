import bisect

max_n = 1000000
omega = [0] * (max_n + 1)
for i in range(2, max_n + 1):
	if omega[i] == 0:
		for j in range(i, max_n + 1, i):
			omega[j] += 1

N_list = []
for i in range(2, max_n + 1):
	if omega[i] == 2:
		N_list.append(i * i)

import sys
data = sys.stdin.read().split()
q = int(data[0])
A_list = list(map(int, data[1:1 + q]))
for A in A_list:
	idx = bisect.bisect_right(N_list, A) - 1
	print(N_list[idx])
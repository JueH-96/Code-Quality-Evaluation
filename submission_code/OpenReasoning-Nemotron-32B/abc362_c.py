n = int(input().strip())
L_list = []
R_list = []
total_min = 0
total_max = 0
for _ in range(n):
	L, R = map(int, input().split())
	L_list.append(L)
	R_list.append(R)
	total_min += L
	total_max += R

if total_min > 0 or total_max < 0:
	print("No")
else:
	D = -total_min
	X = L_list.copy()
	rem = D
	for i in range(n):
		if rem <= 0:
			break
		add_i = min(rem, R_list[i] - L_list[i])
		X[i] += add_i
		rem -= add_i
	print("Yes")
	print(' '.join(map(str, X)))
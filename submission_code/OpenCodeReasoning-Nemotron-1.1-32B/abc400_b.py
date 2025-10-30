data = input().split()
N = int(data[0])
M = int(data[1])

total = 0
term = 1
for i in range(M + 1):
	total += term
	if total > 10**9:
		print("inf")
		break
	if i < M:
		term *= N
else:
	print(total)
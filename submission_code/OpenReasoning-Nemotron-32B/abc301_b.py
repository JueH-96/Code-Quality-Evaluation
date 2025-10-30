n = int(input().strip())
A = list(map(int, input().split()))

res = [A[0]]
for i in range(1, n):
	if res[-1] < A[i]:
		res.extend(range(res[-1] + 1, A[i]))
	elif res[-1] > A[i]:
		res.extend(range(res[-1] - 1, A[i], -1))
	res.append(A[i])

print(" ".join(map(str, res)))
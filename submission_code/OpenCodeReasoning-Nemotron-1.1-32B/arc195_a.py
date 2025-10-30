n, m = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# leftmost matching
j = 0
left_indices = []
for i in range(n):
	if j < m and A[i] == B[j]:
		left_indices.append(i)
		j += 1
if j != m:
	print("No")
	exit(0)

# rightmost matching
j = m - 1
right_indices = []
for i in range(n-1, -1, -1):
	if j >= 0 and A[i] == B[j]:
		right_indices.append(i)
		j -= 1
right_indices.reverse()

if left_indices == right_indices:
	print("No")
else:
	print("Yes")
n = int(input())
A = list(map(int, input().split()))

if A[0] > A[1]:
	max1 = A[0]
	idx1 = 0
	max2 = A[1]
	idx2 = 1
else:
	max1 = A[1]
	idx1 = 1
	max2 = A[0]
	idx2 = 0

for i in range(2, n):
	if A[i] > max1:
		max2 = max1
		idx2 = idx1
		max1 = A[i]
		idx1 = i
	elif A[i] > max2:
		max2 = A[i]
		idx2 = i

print(idx2 + 1)
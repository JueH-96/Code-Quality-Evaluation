n = int(input().strip())
A = list(map(int, input().split()))

pos = [0] * (n + 1)
for idx, num in enumerate(A):
	pos[num] = idx

swaps = []

for i in range(n):
	if A[i] == i + 1:
		continue
		
	j = pos[i + 1]
	x = A[i]
	A[i], A[j] = A[j], A[i]
	
	pos[i + 1] = i
	pos[x] = j
	
	swaps.append((i + 1, j + 1))

print(len(swaps))
for a, b in swaps:
	print(a, b)
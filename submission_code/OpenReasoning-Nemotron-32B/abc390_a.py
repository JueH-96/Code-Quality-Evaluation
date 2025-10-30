target = [1, 2, 3, 4, 5]
A = list(map(int, input().split()))
if A == target:
	print("No")
else:
	for i in range(4):
		B = A.copy()
		B[i], B[i+1] = B[i+1], B[i]
		if B == target:
			print("Yes")
			break
	else:
		print("No")
Q = int(input().strip())
A = []
for _ in range(Q):
	data = input().split()
	if data[0] == '1':
		x = int(data[1])
		A.append(x)
	elif data[0] == '2':
		k = int(data[1])
		print(A[-k])
data = input().split()
X = int(data[0])
Y = int(data[1])

if (Y > X and (Y - X) <= 2) or (Y < X and (X - Y) <= 3):
	print("Yes")
else:
	print("No")
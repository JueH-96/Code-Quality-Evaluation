data = input().split()
A = int(data[0])
B = int(data[1])
C = int(data[2])
total = A + B + C
found = False

if total % 2 == 0:
	half = total // 2
	if A == half or B == half or C == half:
		found = True

if not found and total % 3 == 0:
	third = total // 3
	if A == third and B == third and C == third:
		found = True

print("Yes" if found else "No")
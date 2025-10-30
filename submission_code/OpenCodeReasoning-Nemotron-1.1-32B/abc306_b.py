data = input().split()
total = 0
for i in range(64):
	if data[i] == '1':
		total |= (1 << i)
print(total)
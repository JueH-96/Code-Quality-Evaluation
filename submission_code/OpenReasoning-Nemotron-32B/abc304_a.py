n = int(input().strip())
data = []
for _ in range(n):
	name, age_str = input().split()
	age = int(age_str)
	data.append((age, name))

min_index = 0
for i in range(1, n):
	if data[i][0] < data[min_index][0]:
		min_index = i

for i in range(n):
	idx = (min_index + i) % n
	print(data[idx][1])
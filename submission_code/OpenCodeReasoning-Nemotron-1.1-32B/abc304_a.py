n = int(input().strip())
people = []
for _ in range(n):
	name, age = input().split()
	people.append((name, int(age)))

min_index = min(range(n), key=lambda i: people[i][1])

for i in range(n):
	idx = (min_index + i) % n
	print(people[idx][0])
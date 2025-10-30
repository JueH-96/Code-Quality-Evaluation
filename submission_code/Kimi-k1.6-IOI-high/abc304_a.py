n = int(input())
people = []
for _ in range(n):
    s, a = input().split()
    people.append((s, int(a)))

min_age = float('inf')
min_index = 0
for i in range(n):
    if people[i][1] < min_age:
        min_age = people[i][1]
        min_index = i

for i in range(n):
    current = (min_index + i) % n
    print(people[current][0])
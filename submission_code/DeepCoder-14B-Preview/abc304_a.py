n = int(input())
people = []
for _ in range(n):
    s, a = input().split()
    a = int(a)
    people.append((s, a))

min_age = min(p[1] for p in people)
min_index = None
for i in range(n):
    if people[i][1] == min_age:
        min_index = i
        break

for i in range(n):
    current_index = (min_index + i) % n
    print(people[current_index][0])
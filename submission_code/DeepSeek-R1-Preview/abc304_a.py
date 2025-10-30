n = int(input())
people = []
for _ in range(n):
    s, a = input().split()
    a = int(a)
    people.append((s, a))

min_age = float('inf')
min_idx = 0
for i in range(n):
    if people[i][1] < min_age:
        min_age = people[i][1]
        min_idx = i

min_pos = min_idx + 1

positions_order = [((min_pos - 1 + k) % n) + 1 for k in range(n)]

for p in positions_order:
    print(people[p - 1][0])
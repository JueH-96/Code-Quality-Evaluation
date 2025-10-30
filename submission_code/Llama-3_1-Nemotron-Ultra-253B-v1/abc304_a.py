n = int(input())
people = []
for _ in range(n):
    s, a = input().split()
    people.append((s, int(a)))

min_age = min(a for s, a in people)
start_index = next(i for i, (s, a) in enumerate(people) if a == min_age)

for i in range(n):
    current = (start_index + i) % n
    print(people[current][0])
n = int(input())
people = []
for _ in range(n):
    s, a = input().split()
    people.append((s, int(a)))

start = min(range(n), key=lambda i: people[i][1])

for i in range(n):
    idx = (start + i) % n
    print(people[idx][0])
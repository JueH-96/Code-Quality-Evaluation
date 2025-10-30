from collections import defaultdict

N, M = map(int, input().split())
noodles = [0] * N
people = defaultdict(list)

for _ in range(M):
    T, W, S = map(int, input().split())
    people[T].append((W, S))

current_time = 0
for t in sorted(people.keys()):
    while current_time < t:
        for i in range(N):
            if people[current_time]:
                w, s = people[current_time].pop(0)
                noodles[i] += w
                current_time += s
            else:
                break
    current_time = t

while current_time <= max(people.keys()):
    for i in range(N):
        if people[current_time]:
            w, s = people[current_time].pop(0)
            noodles[i] += w
            current_time += s
        else:
            break

for n in noodles:
    print(n)
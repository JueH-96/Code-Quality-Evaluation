# YOUR CODE HERE
n, m = map(int, input().split())
if n > 1:
    parents = list(map(int, input().split()))
else:
    input()
insurances = []
for i in range(m):
    x, y = map(int, input().split())
    insurances.append((x, y))

parent_child = [[] for _ in range(n)]
for i, p in enumerate(parents):
    parent_child[p - 1].append(i + 1)

covered = [False] * n
for x, y in insurances:
    x -= 1
    q = [x]
    covered_insurance = [x]
    generation = 0
    while q and generation < y:
        next_q = []
        for person in q:
            next_q.extend(parent_child[person])
        q = next_q
        covered_insurance.extend(q)
        generation += 1
    for person in covered_insurance:
        covered[person] = True

print(sum(covered))
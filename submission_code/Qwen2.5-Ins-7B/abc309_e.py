# N, M = map(int, input().split())
# parents = [0] + list(map(int, input().split()))
# insurances = [list(map(int, input().split())) for _ in range(M)]

N, M = 10, 10
parents = [0, 1, 1, 3, 1, 2, 3, 3, 5, 7]
insurances = [[2, 1], [5, 1], [4, 3], [6, 3], [2, 1], [7, 3], [9, 2], [1, 2], [6, 2], [8, 1]]

covered = set()
descendants = [set() for _ in range(N+1)]

for x, y in insurances:
    current = x
    for _ in range(y):
        descendants[current].add(x)
        current = parents[current]

for i in range(1, N+1):
    current = i
    while current != 0:
        if current in descendants[i]:
            covered.add(i)
            break
        current = parents[current]

print(len(covered))
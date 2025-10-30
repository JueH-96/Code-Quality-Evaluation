# YOUR CODE HERE

N, M = map(int, input().split())
babies = []

for _ in range(M):
    A, B = map(int, input().split())
    babies.append((A, B))

babies.sort()

for A, B in babies:
    if B == 'M':
        print('Yes')
    else:
        print('No')
covered = [[False for _ in range(100)] for _ in range(100)]

N = int(input())
for _ in range(N):
    A, B, C, D = map(int, input().split())
    for x in range(A, B):
        for y in range(C, D):
            covered[x][y] = True

total = 0
for row in covered:
    total += sum(row)

print(total)
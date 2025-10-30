n = int(input())
p = list(map(int, input().split()))

A = []
for i in range(1, n + 1):
    # P_i is 1-indexed, but Python lists are 0-indexed
    A.insert(p[i - 1] - 1, i)

print(' '.join(map(str, A)))
N, M = map(int, input().split())
C = input().split()
D = input().split()
P = list(map(int, input().split()))
total = 0
for c in C:
    if c in D:
        total += P[D.index(c)+1]
    else:
        total += P[0]
print(total)
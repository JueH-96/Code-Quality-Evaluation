import itertools

A = list(map(int, input().split()))
for p in itertools.permutations(A):
    if p[0] * p[1] == p[2]:
        print("Yes")
        break
else:
    print("No")
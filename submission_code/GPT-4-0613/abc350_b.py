N, Q = map(int, input().split())
T = list(map(int, input().split()))

teeth = [1]*N
for t in T:
    teeth[t-1] = 1 - teeth[t-1]

print(sum(teeth))
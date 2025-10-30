N, A = map(int, input().split())
T = list(map(int, input().split()))
F_prev = 0
for t in T:
    F_i = max(t, F_prev) + A
    print(F_i)
    F_prev = F_i
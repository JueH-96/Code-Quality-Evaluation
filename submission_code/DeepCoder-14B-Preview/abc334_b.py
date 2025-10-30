A, M, L, R = map(int, input().split())

D = L - A
E = R - A

k_min = (D + M - 1) // M
k_max = E // M

if k_min > k_max:
    print(0)
else:
    print(k_max - k_min + 1)
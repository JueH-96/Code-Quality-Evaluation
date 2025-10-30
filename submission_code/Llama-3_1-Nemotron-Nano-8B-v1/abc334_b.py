A, M, L, R = map(int, input().split())
lower = L - A
upper = R - A
k_min = (lower + M - 1) // M
k_max = upper // M
if k_max >= k_min:
    print(k_max - k_min + 1)
else:
    print(0)
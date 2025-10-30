A, M, L, R = map(int, input().split())

a = L - A
k_min = (a + M - 1) // M

b = R - A
k_max = b // M

if k_max >= k_min:
    print(k_max - k_min + 1)
else:
    print(0)
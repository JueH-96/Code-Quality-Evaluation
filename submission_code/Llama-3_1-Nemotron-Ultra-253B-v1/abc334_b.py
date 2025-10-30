A, M, L, R = map(int, input().split())

numerator_low = L - A
k_min = (numerator_low + M - 1) // M

numerator_high = R - A
k_max = numerator_high // M

if k_min > k_max:
    print(0)
else:
    print(k_max - k_min + 1)
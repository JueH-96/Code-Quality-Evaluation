A, M, L, R = map(int, input().split())

delta_L = L - A
delta_R = R - A

lower_k = (delta_L + M - 1) // M
upper_k = delta_R // M

if upper_k >= lower_k:
    print(upper_k - lower_k + 1)
else:
    print(0)
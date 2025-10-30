A, M, L, R = map(int, input().split())

x_low = L - A
x_high = R - A

lower = (x_low + M - 1) // M
upper = x_high // M

ans = max(0, upper - lower + 1)
print(ans)
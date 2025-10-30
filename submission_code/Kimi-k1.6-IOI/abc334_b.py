A, M, L, R = map(int, input().split())
x1 = L - A
x2 = R - A
lower = (x1 + M - 1) // M
upper = x2 // M
print(max(0, upper - lower + 1) if upper >= lower else 0)
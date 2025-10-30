import sys
data = sys.stdin.read().strip()
A, M, L, R = map(int, data.split())
div_lower = L - A
div_upper = R - A
k_min = (div_lower + M - 1) // M
k_max = div_upper // M
count = max(0, k_max - k_min + 1)
print(count)
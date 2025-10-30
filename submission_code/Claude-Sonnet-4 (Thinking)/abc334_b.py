import math

A, M, L, R = map(int, input().split())

k_min = math.ceil((L - A) / M)
k_max = math.floor((R - A) / M)

print(max(0, k_max - k_min + 1))
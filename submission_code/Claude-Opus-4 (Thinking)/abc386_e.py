from itertools import combinations

N, K = map(int, input().split())
A = list(map(int, input().split()))

max_xor = 0
for combo in combinations(A, K):
    xor_value = 0
    for x in combo:
        xor_value ^= x
    max_xor = max(max_xor, xor_value)

print(max_xor)
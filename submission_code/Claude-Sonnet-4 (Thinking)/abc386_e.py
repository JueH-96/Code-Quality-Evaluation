from itertools import combinations

n, k = map(int, input().split())
a = list(map(int, input().split()))

max_xor = 0
for combo in combinations(range(n), k):
    xor_val = 0
    for i in combo:
        xor_val ^= a[i]
    max_xor = max(max_xor, xor_val)

print(max_xor)
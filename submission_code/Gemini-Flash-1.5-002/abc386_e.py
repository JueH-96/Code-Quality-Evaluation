from itertools import combinations

N, K = map(int, input().split())
A = list(map(int, input().split()))

max_xor = 0
for combo in combinations(A, K):
    xor_sum = 0
    for num in combo:
        xor_sum ^= num
    max_xor = max(max_xor, xor_sum)

print(max_xor)
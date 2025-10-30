import itertools

n, k = map(int, input().split())
a = list(map(int, input().split()))
max_xor = 0
for comb in itertools.combinations(a, k):
    current = 0
    for num in comb:
        current ^= num
    if current > max_xor:
        max_xor = current
print(max_xor)
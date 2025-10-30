import itertools

n, k = map(int, input().split())
a = list(map(int, input().split()))

total_xor = 0
for num in a:
    total_xor ^= num

m = min(k, n - k)
max_xor = 0

for comb in itertools.combinations(a, m):
    current_xor = 0
    for num in comb:
        current_xor ^= num
    if k != m:
        current_xor = total_xor ^ current_xor
    if current_xor > max_xor:
        max_xor = current_xor

print(max_xor)
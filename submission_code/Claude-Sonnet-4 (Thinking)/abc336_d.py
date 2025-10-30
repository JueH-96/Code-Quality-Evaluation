n = int(input())
a = list(map(int, input().split()))

# Precompute left[i]: max j such that a[i-j'] >= j'+1 for all j' in [0, j-1]
left = [0] * n
for i in range(n):
    j = 0
    while i - j >= 0 and a[i - j] >= j + 1:
        j += 1
    left[i] = j

# Precompute right[i]: max j such that a[i+j'] >= j'+1 for all j' in [0, j-1]
right = [0] * n
for i in range(n):
    j = 0
    while i + j < n and a[i + j] >= j + 1:
        j += 1
    right[i] = j

max_k = 0
for c in range(n):
    k = min(left[c], right[c])
    max_k = max(max_k, k)

print(max_k)
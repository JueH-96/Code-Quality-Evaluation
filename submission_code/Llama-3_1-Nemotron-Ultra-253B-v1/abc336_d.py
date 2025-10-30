n = int(input())
a = list(map(int, input().split()))

prefix_min = [0] * n
prefix_min[0] = a[0] - 0
for i in range(1, n):
    prefix_min[i] = min(prefix_min[i-1], a[i] - i)

suffix_min = [0] * n
suffix_min[-1] = a[-1] + (n-1)
for i in range(n-2, -1, -1):
    suffix_min[i] = min(suffix_min[i+1], a[i] + i)

max_k = 0
for m in range(n):
    min_left = m + prefix_min[m]
    min_right = suffix_min[m] - m
    k_candidate = min(min_left, min_right)
    k_max_candidate = min(m + 1, n - m)
    current_k = min(k_candidate, k_max_candidate)
    if current_k > max_k:
        max_k = current_k

print(max_k)
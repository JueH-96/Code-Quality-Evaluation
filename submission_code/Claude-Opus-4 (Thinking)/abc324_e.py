def max_prefix_len(s, t):
    j = 0
    for c in s:
        if j < len(t) and c == t[j]:
            j += 1
    return j

def is_suffix_subsequence(s, t, k):
    j = k
    for c in s:
        if j < len(t) and c == t[j]:
            j += 1
    return j == len(t)

def compute_min_suffix(s, t):
    n = len(t)
    left, right = 0, n + 1
    while left < right:
        mid = (left + right) // 2
        if is_suffix_subsequence(s, t, mid):
            right = mid
        else:
            left = mid + 1
    return left

n, t = input().split()
n = int(n)
strings = [input().strip() for _ in range(n)]

prefix_len = [max_prefix_len(s, t) for s in strings]
min_suffix = [compute_min_suffix(s, t) for s in strings]

count = 0
for i in range(n):
    for j in range(n):
        if prefix_len[i] >= min_suffix[j]:
            count += 1

print(count)
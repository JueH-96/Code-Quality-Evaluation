import sys

def max_gift_value(n, m, d, a, b):
    a.sort()
    b.sort()
    max_value = -1
    i = 0
    j = 0
    while i < n and j < m:
        if abs(a[i] - b[j]) <= d:
            max_value = max(max_value, a[i] + b[j])
            i += 1
            j += 1
        elif a[i] - b[j] > d:
            j += 1
        else:
            i += 1
    return max_value

n, m, d = map(int, sys.stdin.readline().split())
a = list(map(int, sys.stdin.readline().split()))
b = list(map(int, sys.stdin.readline().split()))

print(max_gift_value(n, m, d, a, b))
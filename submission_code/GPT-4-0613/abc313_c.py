import sys
from collections import Counter

def min_operations():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    a.sort()
    c = Counter(a)
    keys = sorted(c.keys())
    ans = 0
    for i in range(len(keys) - 1):
        if keys[i] + 1 < keys[i + 1]:
            ans += c[keys[i]]
            c[keys[i] + 1] += c[keys[i]]
        elif keys[i] + 1 == keys[i + 1]:
            ans += min(c[keys[i]], c[keys[i + 1]])
            c[keys[i + 1]] += max(0, c[keys[i]] - c[keys[i + 1]])
        else:
            ans += c[keys[i + 1]]
            c[keys[i]] += c[keys[i + 1]]
    print(ans)

min_operations()
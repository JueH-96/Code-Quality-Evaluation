import math

def gcd(a, b):
    if b == 0:
        return a
    return gcd(b, a % b)

def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))

    for i in range(n):
        max_gcd = 0
        for j in range(1 << n):
            subset = []
            for l in range(n):
                if (j >> l) & 1:
                    subset.append(a[l])
            
            if len(subset) == k and a[i] in subset:
                current_gcd = 0
                if len(subset) > 0:
                    current_gcd = subset[0]
                    for m in range(1, len(subset)):
                        current_gcd = gcd(current_gcd, subset[m])
                max_gcd = max(max_gcd, current_gcd)
        print(max_gcd)

solve()
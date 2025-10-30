def max_product(n, a):
    a.sort()
    return max(a[0]*a[1]*a[n-1], a[n-3]*a[n-2]*a[n-1])

t = int(input().strip())
for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    print(max_product(n, a))
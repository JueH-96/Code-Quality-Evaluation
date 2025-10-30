n = int(input())
a = list(map(int, input().split()))

for i in range(n - 1):
    s, t = map(int, input().split())
    conversions = a[i] // s
    a[i] -= conversions * s
    a[i + 1] += conversions * t

print(a[n - 1])
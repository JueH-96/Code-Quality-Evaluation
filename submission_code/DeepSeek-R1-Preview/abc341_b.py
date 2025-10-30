n = int(input())
a = list(map(int, input().split()))
st = [tuple(map(int, input().split())) for _ in range(n-1)]

for i in range(n-1):
    s, t = st[i]
    count = a[i] // s
    a[i] -= count * s
    a[i+1] += count * t

print(a[-1])
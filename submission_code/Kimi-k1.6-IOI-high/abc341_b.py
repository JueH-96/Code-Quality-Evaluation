n = int(input())
a = list(map(int, input().split()))
s = []
t = []
for _ in range(n-1):
    si, ti = map(int, input().split())
    s.append(si)
    t.append(ti)

for i in range(n-1):
    current = a[i]
    conv = current // s[i]
    a[i+1] += conv * t[i]
    a[i] = current % s[i]

print(a[-1])
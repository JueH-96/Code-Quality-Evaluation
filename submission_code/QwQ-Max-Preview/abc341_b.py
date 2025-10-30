n = int(input())
a = list(map(int, input().split()))
s = []
t = []
for _ in range(n-1):
    si, ti = map(int, input().split())
    s.append(si)
    t.append(ti)

# Process left to right
for i in range(n-1):
    current_s = s[i]
    current_t = t[i]
    count = a[i] // current_s
    a[i] -= count * current_s
    a[i+1] += count * current_t

# Process right to left
for i in range(n-2, -1, -1):
    current_s = s[i]
    current_t = t[i]
    count = a[i] // current_s
    a[i] -= count * current_s
    a[i+1] += count * current_t

print(a[-1])
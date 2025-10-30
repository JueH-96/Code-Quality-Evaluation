n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

a_set = set(a)
merged = a + b
c = sorted(merged)

found = False
for i in range(len(c) - 1):
    if c[i] in a_set and c[i+1] in a_set:
        found = True
        break

print("Yes" if found else "No")
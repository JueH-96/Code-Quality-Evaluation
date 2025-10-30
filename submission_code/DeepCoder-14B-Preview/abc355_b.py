n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

combined = a + b
combined.sort()

a_set = set(a)

found = False
for i in range(len(combined) - 1):
    if combined[i] in a_set and combined[i+1] in a_set:
        found = True
        break

print("Yes" if found else "No")
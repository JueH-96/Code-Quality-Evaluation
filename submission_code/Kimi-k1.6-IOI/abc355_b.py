n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

combined = []
for num in a:
    combined.append((num, True))
for num in b:
    combined.append((num, False))

combined.sort(key=lambda x: x[0])

found = False
for i in range(len(combined) - 1):
    if combined[i][1] and combined[i+1][1]:
        found = True
        break

print("Yes" if found else "No")
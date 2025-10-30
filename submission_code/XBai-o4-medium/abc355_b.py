n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

combined = []
for num in a:
    combined.append((num, 'A'))
for num in b:
    combined.append((num, 'B'))

sorted_combined = sorted(combined, key=lambda x: x[0])

found = False
for i in range(len(sorted_combined) - 1):
    if sorted_combined[i][1] == 'A' and sorted_combined[i+1][1] == 'A':
        found = True
        break

print("Yes" if found else "No")
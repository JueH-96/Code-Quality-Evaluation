from collections import Counter

a, b, c, d = map(int, input().split())
counts = Counter([a, b, c, d])

possible = False
for x in range(1, 14):
    temp = counts.copy()
    temp[x] += 1
    sorted_vals = sorted(temp.values(), reverse=True)
    if sorted_vals == [3, 2]:
        possible = True
        break

print("Yes" if possible else "No")
n, m = map(int, input().split())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

combined = a + b
combined_sorted = sorted(combined)
a_set = set(a)

found = False
for i in range(len(combined_sorted) - 1):
    current = combined_sorted[i]
    next_num = combined_sorted[i+1]
    if current in a_set and next_num in a_set:
        found = True
        break

print("Yes" if found else "No")
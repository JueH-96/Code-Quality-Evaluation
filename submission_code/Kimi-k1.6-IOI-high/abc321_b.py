n, x = map(int, input().split())
a = list(map(int, input().split()))
found = False

for s in range(0, 101):
    combined = a + [s]
    combined_sorted = sorted(combined)
    total = sum(combined_sorted[1 : n-1])
    if total >= x:
        print(s)
        found = True
        break

if not found:
    print(-1)
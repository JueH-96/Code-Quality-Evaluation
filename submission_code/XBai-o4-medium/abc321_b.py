n, x = map(int, input().split())
a = list(map(int, input().split()))

for s in range(0, 101):
    combined = a + [s]
    combined_sorted = sorted(combined)
    middle = combined_sorted[1:-1]
    if sum(middle) >= x:
        print(s)
        exit()
print(-1)
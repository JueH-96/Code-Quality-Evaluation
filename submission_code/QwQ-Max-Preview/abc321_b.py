n, x = map(int, input().split())
a = list(map(int, input().split()))
min_y = 101  # Initialize with a value higher than possible Y

for y in range(0, 101):
    combined = a + [y]
    combined_sorted = sorted(combined)
    total = sum(combined_sorted[1:n-1])  # Sum from index 1 to n-2 inclusive (slice 1:n-1)
    if total >= x:
        if y < min_y:
            min_y = y

print(min_y if min_y != 101 else -1)
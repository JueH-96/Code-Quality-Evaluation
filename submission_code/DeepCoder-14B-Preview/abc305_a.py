n = int(input())
stations = [5 * i for i in range(21)]
closest = None
min_diff = float('inf')

for s in stations:
    diff = abs(n - s)
    if diff < min_diff:
        min_diff = diff
        closest = s

print(closest)
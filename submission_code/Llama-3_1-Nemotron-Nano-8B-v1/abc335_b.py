n = int(input())
for x in range(n + 1):
    for y in range(n - x + 1):
        remaining = n - x - y
        for z in range(remaining + 1):
            print(x, y, z)
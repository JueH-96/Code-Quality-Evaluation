n = int(input())
for x in range(n + 1):
    for y in range(n + 1 - x):
        for z in range(n + 1 - x - y):
            print(x, y, z)
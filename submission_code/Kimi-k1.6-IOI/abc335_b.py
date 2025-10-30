n = int(input())
for x in range(n + 1):
    for y in range(n - x + 1):
        max_z = n - x - y
        for z in range(max_z + 1):
            print(f"{x} {y} {z}")
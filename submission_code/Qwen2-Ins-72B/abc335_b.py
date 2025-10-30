n = int(input())

for x in range(n+1):
    for y in range(n+1):
        z_max = n - x - y
        for z in range(z_max+1):
            print(x, y, z)
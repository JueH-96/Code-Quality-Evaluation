N = int(input().strip())
for x in range(N+1):
    for y in range(N+1-x):
        for z in range(N+1-x-y):
            print(x, y, z)
# YOUR CODE HERE
v1, v2, v3 = map(int, input().split())

for x in range(1, 8):
    for y in range(1, 8):
        for z in range(1, 8):
            if x * y * z == v3 and x * y * (7 - z) * 2 + x * (7 - y) * z * 2 + (7 - x) * y * z * 2 == v2 and (7 - x) * (7 - y) * (7 - z) * 3 + (7 - x) * (7 - y) * z + (7 - x) * y * (7 - z) + x * (7 - y) * (7 - z) + x * y * (7 - z) + x * (7 - y) * z + (7 - x) * y * z == v1:
                print("Yes")
                print(0, 0, 0, 0, 7 - y, 0, 7 - x, 0, 0)
                exit()

print("No")
def solve():
    n = int(input())
    cuboids = []
    for _ in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids.append(tuple(sorted((x1, x2)) + sorted((y1, y2)) + sorted((z1, z2))))

    ans = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            
            x1i, x2i, y1i, y2i, z1i, z2i = cuboids[i]
            x1j, x2j, y1j, y2j, z1j, z2j = cuboids[j]

            
            intersect_x = max(0, min(x2i, x2j) - max(x1i, x1j))
            intersect_y = max(0, min(y2i, y2j) - max(y1i, y1j))
            intersect_z = max(0, min(z2i, z2j) - max(z1i, z1j))

            
            if (intersect_x * intersect_y > 0 and intersect_z == 0) or \
               (intersect_x * intersect_z > 0 and intersect_y == 0) or \
               (intersect_y * intersect_z > 0 and intersect_x == 0):
                ans[i] += 1

    for count in ans:
        print(count)

solve()
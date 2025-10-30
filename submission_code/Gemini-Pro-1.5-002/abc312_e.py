# YOUR CODE HERE
def solve():
    n = int(input())
    cuboids = []
    for _ in range(n):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids.append(((x1, y1, z1), (x2, y2, z2)))

    ans = []
    for i in range(n):
        count = 0
        for j in range(n):
            if i == j:
                continue

            cuboid1 = cuboids[i]
            cuboid2 = cuboids[j]

            x11, y11, z11 = cuboid1[0]
            x12, y12, z12 = cuboid1[1]
            x21, y21, z21 = cuboid2[0]
            x22, y22, z22 = cuboid2[1]
            
            if (x12 == x21 or x22 == x11) and (y11 <= y22 and y21 <= y12) and (z11 <= z22 and z21 <= z12):
                count +=1
            elif (y12 == y21 or y22 == y11) and (x11 <= x22 and x21 <= x12) and (z11 <= z22 and z21 <= z12):
                count += 1
            elif (z12 == z21 or z22 == z11) and (x11 <= x22 and x21 <= x12) and (y11 <= y22 and y21 <= y12):
                count += 1
                
        ans.append(count)

    for c in ans:
        print(c)

solve()
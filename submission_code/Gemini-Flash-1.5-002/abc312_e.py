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
            
            (x1_i, y1_i, z1_i), (x2_i, y2_i, z2_i) = cuboids[i]
            (x1_j, y1_j, z1_j), (x2_j, y2_j, z2_j) = cuboids[j]

            shared_face = False
            
            #Check for shared faces
            if (x1_i == x2_j and x2_i > x2_j and 
                max(y1_i, y1_j) < min(y2_i, y2_j) and
                max(z1_i, z1_j) < min(z2_i, z2_j)):
                shared_face = True
            if (x2_i == x1_j and x1_i < x1_j and
                max(y1_i, y1_j) < min(y2_i, y2_j) and
                max(z1_i, z1_j) < min(z2_i, z2_j)):
                shared_face = True
            if (y1_i == y2_j and y2_i > y2_j and
                max(x1_i, x1_j) < min(x2_i, x2_j) and
                max(z1_i, z1_j) < min(z2_i, z2_j)):
                shared_face = True
            if (y2_i == y1_j and y1_i < y1_j and
                max(x1_i, x1_j) < min(x2_i, x2_j) and
                max(z1_i, z1_j) < min(z2_i, z2_j)):
                shared_face = True
            if (z1_i == z2_j and z2_i > z2_j and
                max(x1_i, x1_j) < min(x2_i, x2_j) and
                max(y1_i, y1_j) < min(y2_i, y2_j)):
                shared_face = True
            if (z2_i == z1_j and z1_i < z1_j and
                max(x1_i, x1_j) < min(x2_i, x2_j) and
                max(y1_i, y1_j) < min(y2_i, y2_j)):
                shared_face = True

            if shared_face:
                count += 1
        ans.append(count)
    print(*ans)

solve()
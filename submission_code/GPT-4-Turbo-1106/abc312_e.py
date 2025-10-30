def count_shared_faces(cuboids):
    n = len(cuboids)
    shared_faces = [0] * n

    # Check each pair of cuboids to see if they share a face
    for i in range(n):
        for j in range(i + 1, n):
            # Unpack the coordinates for easier comparison
            xi1, yi1, zi1, xi2, yi2, zi2 = cuboids[i]
            xj1, yj1, zj1, xj2, yj2, zj2 = cuboids[j]

            # Check if the cuboids share a face on the x, y, or z plane
            if (xi2 == xj1 and yi1 < yj2 and yi2 > yj1 and zi1 < zj2 and zi2 > zj1) or \
               (xi1 == xj2 and yi1 < yj2 and yi2 > yj1 and zi1 < zj2 and zi2 > zj1) or \
               (yi2 == yj1 and xi1 < xj2 and xi2 > xj1 and zi1 < zj2 and zi2 > zj1) or \
               (yi1 == yj2 and xi1 < xj2 and xi2 > xj1 and zi1 < zj2 and zi2 > zj1) or \
               (zi2 == zj1 and xi1 < xj2 and xi2 > xj1 and yi1 < yj2 and yi2 > yj1) or \
               (zi1 == zj2 and xi1 < xj2 and xi2 > xj1 and yi1 < yj2 and yi2 > yj1):
                shared_faces[i] += 1
                shared_faces[j] += 1

    return shared_faces

# Read input
N = int(input().strip())
cuboids = [list(map(int, input().strip().split())) for _ in range(N)]

# Solve the problem
result = count_shared_faces(cuboids)

# Output the result
for count in result:
    print(count)
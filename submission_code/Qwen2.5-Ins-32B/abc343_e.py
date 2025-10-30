import sys

def volume_of_intersection(cube1, cube2):
    x1, y1, z1 = cube1
    x2, y2, z2 = cube2
    dx = max(0, min(x1+7, x2+7) - max(x1, x2))
    dy = max(0, min(y1+7, y2+7) - max(y1, y2))
    dz = max(0, min(z1+7, z2+7) - max(z1, z2))
    return dx * dy * dz

def volume_of_union(cube1, cube2, cube3):
    v1 = volume_of_intersection(cube1, cube2)
    v2 = volume_of_intersection(cube2, cube3)
    v3 = volume_of_intersection(cube3, cube1)
    v123 = volume_of_intersection(cube1, cube2)
    v123 = volume_of_intersection((0, 0, 0), (v123, v123, v123))
    return v1 + v2 + v3 - 2 * v123

def solve(V1, V2, V3):
    for a1 in range(-100, 101):
        for b1 in range(-100, 101):
            for c1 in range(-100, 101):
                for a2 in range(-100, 101):
                    for b2 in range(-100, 101):
                        for c2 in range(-100, 101):
                            for a3 in range(-100, 101):
                                for b3 in range(-100, 101):
                                    for c3 in range(-100, 101):
                                        cube1 = (a1, b1, c1)
                                        cube2 = (a2, b2, c2)
                                        cube3 = (a3, b3, c3)
                                        v1 = 3 * 7**3 - volume_of_union(cube1, cube2, cube3)
                                        v2 = volume_of_intersection(cube1, cube2) + volume_of_intersection(cube2, cube3) + volume_of_intersection(cube3, cube1) - 3 * volume_of_intersection(cube1, cube2, cube3)
                                        v3 = volume_of_intersection(cube1, cube2, cube3)
                                        if v1 == V1 and v2 == V2 and v3 == V3:
                                            print("Yes")
                                            print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                            return
    print("No")

V1, V2, V3 = map(int, input().split())
solve(V1, V2, V3)
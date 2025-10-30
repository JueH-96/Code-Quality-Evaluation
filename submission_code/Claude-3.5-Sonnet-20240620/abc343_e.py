# YOUR CODE HERE
import itertools

def volume_intersection(cube1, cube2):
    x1, y1, z1 = cube1
    x2, y2, z2 = cube2
    dx = max(0, min(x1+7, x2+7) - max(x1, x2))
    dy = max(0, min(y1+7, y2+7) - max(y1, y2))
    dz = max(0, min(z1+7, z2+7) - max(z1, z2))
    return dx * dy * dz

def volume_union(cube1, cube2, cube3):
    v1 = 7**3 * 3
    v2 = volume_intersection(cube1, cube2) + volume_intersection(cube2, cube3) + volume_intersection(cube3, cube1)
    v3 = volume_intersection(cube1, cube2, cube3)
    return v1 - v2 + v3

def volume_intersection(cube1, cube2, cube3):
    x1, y1, z1 = cube1
    x2, y2, z2 = cube2
    x3, y3, z3 = cube3
    dx = max(0, min(x1+7, x2+7, x3+7) - max(x1, x2, x3))
    dy = max(0, min(y1+7, y2+7, y3+7) - max(y1, y2, y3))
    dz = max(0, min(z1+7, z2+7, z3+7) - max(z1, z2, z3))
    return dx * dy * dz

def solve(V1, V2, V3):
    for a1, b1, c1, a2, b2, c2, a3, b3, c3 in itertools.product(range(-100, 101), repeat=9):
        cube1 = (a1, b1, c1)
        cube2 = (a2, b2, c2)
        cube3 = (a3, b3, c3)
        
        v3 = volume_intersection(cube1, cube2, cube3)
        if v3 != V3:
            continue
        
        v2 = (volume_intersection(cube1, cube2) + 
              volume_intersection(cube2, cube3) + 
              volume_intersection(cube3, cube1) - 3*v3)
        if v2 != V2:
            continue
        
        v1 = volume_union(cube1, cube2, cube3) - v2 - v3
        if v1 == V1:
            return f"Yes
{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}"
    
    return "No"

V1, V2, V3 = map(int, input().split())
print(solve(V1, V2, V3))
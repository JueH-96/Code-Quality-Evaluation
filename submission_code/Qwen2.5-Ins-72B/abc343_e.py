# YOUR CODE HERE
def calculate_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    def volume(x1, y1, z1, x2, y2, z2):
        return max(0, x2 - x1) * max(0, y2 - y1) * max(0, z2 - z1)
    
    V1 = 0
    V2 = 0
    V3 = 0
    
    # Volume of each cube
    V1 += volume(a1, b1, c1, a1 + 7, b1 + 7, c1 + 7)
    V1 += volume(a2, b2, c2, a2 + 7, b2 + 7, c2 + 7)
    V1 += volume(a3, b3, c3, a3 + 7, b3 + 7, c3 + 7)
    
    # Volume of intersections of two cubes
    V2 += volume(max(a1, a2), max(b1, b2), max(c1, c2), min(a1 + 7, a2 + 7), min(b1 + 7, b2 + 7), min(c1 + 7, c2 + 7))
    V2 += volume(max(a1, a3), max(b1, b3), max(c1, c3), min(a1 + 7, a3 + 7), min(b1 + 7, b3 + 7), min(c1 + 7, c3 + 7))
    V2 += volume(max(a2, a3), max(b2, b3), max(c2, c3), min(a2 + 7, a3 + 7), min(b2 + 7, b3 + 7), min(c2 + 7, c3 + 7))
    
    # Volume of intersection of all three cubes
    V3 = volume(max(a1, a2, a3), max(b1, b2, b3), max(c1, c2, c3), min(a1 + 7, a2 + 7, a3 + 7), min(b1 + 7, b2 + 7, b3 + 7), min(c1 + 7, c2 + 7, c3 + 7))
    
    # Adjust V1 and V2 to account for overlaps
    V1 -= 2 * V2
    V1 -= 3 * V3
    V2 -= 3 * V3
    
    return V1, V2, V3

def find_cubes(V1, V2, V3):
    for a1 in range(-100, 101):
        for b1 in range(-100, 101):
            for c1 in range(-100, 101):
                for a2 in range(-100, 101):
                    for b2 in range(-100, 101):
                        for c2 in range(-100, 101):
                            for a3 in range(-100, 101):
                                for b3 in range(-100, 101):
                                    for c3 in range(-100, 101):
                                        v1, v2, v3 = calculate_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3)
                                        if v1 == V1 and v2 == V2 and v3 == V3:
                                            return a1, b1, c1, a2, b2, c2, a3, b3, c3
    return None

V1, V2, V3 = map(int, input().split())
result = find_cubes(V1, V2, V3)
if result:
    print("Yes")
    print(" ".join(map(str, result)))
else:
    print("No")
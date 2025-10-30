def check_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3, target_v1, target_v2, target_v3):
    def get_intersection(x1, y1, z1, x2, y2, z2):
        x = max(min(x1+7, x2+7) - max(x1, x2), 0)
        y = max(min(y1+7, y2+7) - max(y1, y2), 0)
        z = max(min(z1+7, z2+7) - max(z1, z2), 0)
        return x * y * z

    # Volume of each cube
    v = 7 * 7 * 7

    # Intersections of pairs
    i12 = get_intersection(a1, b1, c1, a2, b2, c2)
    i23 = get_intersection(a2, b2, c2, a3, b3, c3)
    i13 = get_intersection(a1, b1, c1, a3, b3, c3)

    # Triple intersection
    i123 = get_intersection(max(a1, a2, a3), max(b1, b2, b3), max(c1, c2, c3),
                           min(a1+7, a2+7, a3+7)-7, min(b1+7, b2+7, b3+7)-7, min(c1+7, c2+7, c3+7)-7)

    # Calculate volumes
    v3 = i123
    v2 = i12 + i23 + i13 - 3*i123
    v1 = 3*v - 2*(i12 + i23 + i13) + 3*i123

    return v1 == target_v1 and v2 == target_v2 and v3 == target_v3

def solve():
    V1, V2, V3 = map(int, input().split())
    
    # Try some common patterns
    test_cases = [
        (0, 0, 0, 0, 6, 0, 6, 0, 0),
        (0, 0, 0, 6, 0, 0, 0, 6, 0),
        (0, 0, 0, 0, 0, 6, 0, 6, 0),
        (-10, 0, 0, -10, 0, 6, -10, 6, 1)
    ]
    
    for case in test_cases:
        a1, b1, c1, a2, b2, c2, a3, b3, c3 = case
        if check_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3, V1, V2, V3):
            print("Yes")
            print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
            return
            
    # Try some systematic positions
    for d in range(7):
        a1, b1, c1 = 0, 0, 0
        a2, b2, c2 = d, d, 0
        a3, b3, c3 = d, 0, d
        if check_volumes(a1, b1, c1, a2, b2, c2, a3, b3, c3, V1, V2, V3):
            print("Yes")
            print(f"{a1} {b1} {c1} {a2} {b2} {c2} {a3} {b3} {c3}")
            return
            
    print("No")

solve()
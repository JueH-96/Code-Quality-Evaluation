def solve(V1, V2, V3):
    # Check necessary condition
    if V1 + 2*V2 + 3*V3 != 1029:
        return None
    
    def intersection_volume(cube1, cube2):
        a1, b1, c1 = cube1
        a2, b2, c2 = cube2
        x_overlap = max(0, 7 - abs(a1 - a2))
        y_overlap = max(0, 7 - abs(b1 - b2))
        z_overlap = max(0, 7 - abs(c1 - c2))
        return x_overlap * y_overlap * z_overlap
    
    def triple_intersection_volume(cube1, cube2, cube3):
        a1, b1, c1 = cube1
        a2, b2, c2 = cube2
        a3, b3, c3 = cube3
        x_overlap = max(0, min(a1+7, a2+7, a3+7) - max(a1, a2, a3))
        y_overlap = max(0, min(b1+7, b2+7, b3+7) - max(b1, b2, b3))
        z_overlap = max(0, min(c1+7, c2+7, c3+7) - max(c1, c2, c3))
        return x_overlap * y_overlap * z_overlap
    
    def check_solution(cube1, cube2, cube3):
        I12 = intersection_volume(cube1, cube2)
        I13 = intersection_volume(cube1, cube3)
        I23 = intersection_volume(cube2, cube3)
        I123 = triple_intersection_volume(cube1, cube2, cube3)
        
        calc_V2 = I12 + I13 + I23 - 3*I123
        calc_V3 = I123
        calc_V1 = 1029 - 2*(I12 + I13 + I23) + 3*I123
        
        return calc_V1 == V1 and calc_V2 == V2 and calc_V3 == V3
    
    # Try some specific patterns first
    patterns = [
        # Pattern: C1=(0,0,0), C2=(d,0,0), C3=(0,d,0)
        lambda d: [(0, 0, 0), (d, 0, 0), (0, d, 0)],
        # Pattern: C1=(0,0,0), C2=(0,d,0), C3=(0,0,d)
        lambda d: [(0, 0, 0), (0, d, 0), (0, 0, d)],
        # Pattern: C1=(0,0,0), C2=(d,0,0), C3=(0,0,d)
        lambda d: [(0, 0, 0), (d, 0, 0), (0, 0, d)],
    ]
    
    for pattern in patterns:
        for d in range(-20, 21):
            cubes = pattern(d)
            if all(abs(coord) <= 100 for cube in cubes for coord in cube):
                if check_solution(*cubes):
                    return cubes
    
    # If patterns don't work, try brute force in a small range
    for a1 in range(-5, 6):
        for b1 in range(-5, 6):
            for c1 in range(-5, 6):
                for a2 in range(-5, 6):
                    for b2 in range(-5, 6):
                        for c2 in range(-5, 6):
                            for a3 in range(-5, 6):
                                for b3 in range(-5, 6):
                                    for c3 in range(-5, 6):
                                        cubes = [(a1, b1, c1), (a2, b2, c2), (a3, b3, c3)]
                                        if check_solution(*cubes):
                                            return cubes
    
    return None

V1, V2, V3 = map(int, input().split())
result = solve(V1, V2, V3)

if result is None:
    print("No")
else:
    print("Yes")
    flat = []
    for a, b, c in result:
        flat.extend([a, b, c])
    print(*flat)
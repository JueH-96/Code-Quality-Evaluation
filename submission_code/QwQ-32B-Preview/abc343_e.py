def calculate_overlaps(a1, b1, c1, a2, b2, c2, a3, b3, c3):
    # Calculate pairwise overlaps
    def overlap(cube1, cube2):
        dx = max(0, min(cube1[0]+7, cube2[0]+7) - max(cube1[0], cube2[0]))
        dy = max(0, min(cube1[1]+7, cube2[1]+7) - max(cube1[1], cube2[1]))
        dz = max(0, min(cube1[2]+7, cube2[2]+7) - max(cube1[2], cube2[2]))
        return dx * dy * dz

    o12 = overlap((a1, b1, c1), (a2, b2, c2))
    o13 = overlap((a1, b1, c1), (a3, b3, c3))
    o23 = overlap((a2, b2, c2), (a3, b3, c3))
    
    # Triple overlap
    dx = max(0, min(a1+7, a2+7, a3+7) - max(a1, a2, a3))
    dy = max(0, min(b1+7, b2+7, b3+7) - max(b1, b2, b3))
    dz = max(0, min(c1+7, c2+7, c3+7) - max(c1, c2, c3))
    o123 = dx * dy * dz
    
    # Volumes
    v1 = 3*343 - 2*(o12 + o13 + o23) + 3*o123
    v2 = o12 + o13 + o23 - 3*o123
    v3 = o123
    return v1, v2, v3

def find_positions(v1, v2, v3):
    if v1 + 2*v2 + 3*v3 != 1029:
        return "No", []
    
    # Try sample positions first
    positions = [
        (0, 0, 0, 0, 6, 0, 6, 0, 0),
        (0, 0, 0, 0, 7, 0, 7, 0, 0),
        (-10, 0, 0, -10, 0, 6, -10, 6, 1)
    ]
    
    for pos in positions:
        a1, b1, c1, a2, b2, c2, a3, b3, c3 = pos
        calculated_v1, calculated_v2, calculated_v3 = calculate_overlaps(a1, b1, c1, a2, b2, c2, a3, b3, c3)
        if calculated_v1 == v1 and calculated_v2 == v2 and calculated_v3 == v3:
            return "Yes", pos
    
    # If not found in sample positions, perform a limited search
    for a2 in range(-10, 10):
        for b2 in range(-10, 10):
            for a3 in range(-10, 10):
                for b3 in range(-10, 10):
                    pos = (0, 0, 0, a2, b2, 0, a3, b3, 0)
                    calculated_v1, calculated_v2, calculated_v3 = calculate_overlaps(*pos)
                    if calculated_v1 == v1 and calculated_v2 == v2 and calculated_v3 == v3:
                        return "Yes", pos
    return "No", []

def main():
    import sys
    v1, v2, v3 = map(int, sys.stdin.read().split())
    result, positions = find_positions(v1, v2, v3)
    print(result)
    if result == "Yes":
        print(' '.join(map(str, positions)))

if __name__ == "__main__":
    main()
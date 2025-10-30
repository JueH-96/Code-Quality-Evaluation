def intersection_volume(cube1, cube2):
    """Calculate intersection volume of two cubes"""
    a1, b1, c1 = cube1
    a2, b2, c2 = cube2
    
    x_overlap = max(0, min(a1 + 7, a2 + 7) - max(a1, a2))
    y_overlap = max(0, min(b1 + 7, b2 + 7) - max(b1, b2))
    z_overlap = max(0, min(c1 + 7, c2 + 7) - max(c1, c2))
    
    return x_overlap * y_overlap * z_overlap

def intersection_volume_three(cube1, cube2, cube3):
    """Calculate intersection volume of three cubes"""
    a1, b1, c1 = cube1
    a2, b2, c2 = cube2
    a3, b3, c3 = cube3
    
    x_overlap = max(0, min(a1 + 7, a2 + 7, a3 + 7) - max(a1, a2, a3))
    y_overlap = max(0, min(b1 + 7, b2 + 7, b3 + 7) - max(b1, b2, b3))
    z_overlap = max(0, min(c1 + 7, c2 + 7, c3 + 7) - max(c1, c2, c3))
    
    return x_overlap * y_overlap * z_overlap

def check_volumes(cube1, cube2, cube3, v1, v2, v3):
    """Check if the cube configuration gives the desired volumes"""
    # Calculate intersection volumes
    v12 = intersection_volume(cube1, cube2)
    v13 = intersection_volume(cube1, cube3)
    v23 = intersection_volume(cube2, cube3)
    v123 = intersection_volume_three(cube1, cube2, cube3)
    
    # Calculate volumes in exactly 1, 2, and 3 cubes
    volume_3 = v123
    volume_2 = v12 + v13 + v23 - 3 * v123
    volume_1 = 3 * 343 - 2 * (v12 + v13 + v23) + 3 * v123
    
    return volume_1 == v1 and volume_2 == v2 and volume_3 == v3

# Read input
v1, v2, v3 = map(int, input().split())

# Try different configurations
found = False

# Strategy: Try placing cubes with controlled overlaps
# Start with simple configurations where cubes overlap in predictable ways

# Try configurations where cubes overlap along one axis
for overlap in range(8):  # 0 to 7
    if found:
        break
    
    # Configuration 1: Three cubes in a line along x-axis
    cube1 = (0, 0, 0)
    cube2 = (7 - overlap, 0, 0)
    cube3 = (2 * (7 - overlap), 0, 0)
    
    if all(abs(coord) <= 100 for cube in [cube1, cube2, cube3] for coord in cube):
        if check_volumes(cube1, cube2, cube3, v1, v2, v3):
            print("Yes")
            print(f"{cube1[0]} {cube1[1]} {cube1[2]} {cube2[0]} {cube2[1]} {cube2[2]} {cube3[0]} {cube3[1]} {cube3[2]}")
            found = True
            break

# Try L-shaped configurations
if not found:
    for overlap1 in range(8):
        for overlap2 in range(8):
            if found:
                break
            
            cube1 = (0, 0, 0)
            cube2 = (0, 7 - overlap1, 0)
            cube3 = (7 - overlap2, 0, 0)
            
            if all(abs(coord) <= 100 for cube in [cube1, cube2, cube3] for coord in cube):
                if check_volumes(cube1, cube2, cube3, v1, v2, v3):
                    print("Yes")
                    print(f"{cube1[0]} {cube1[1]} {cube1[2]} {cube2[0]} {cube2[1]} {cube2[2]} {cube3[0]} {cube3[1]} {cube3[2]}")
                    found = True
                    break

# Try more complex configurations with offsets in multiple dimensions
if not found:
    for dx in range(-10, 11):
        for dy in range(-10, 11):
            for dz in range(-10, 11):
                if found:
                    break
                
                cube1 = (0, 0, 0)
                cube2 = (dx, dy, 0)
                cube3 = (dx, 0, dz)
                
                if all(abs(coord) <= 100 for cube in [cube1, cube2, cube3] for coord in cube):
                    if check_volumes(cube1, cube2, cube3, v1, v2, v3):
                        print("Yes")
                        print(f"{cube1[0]} {cube1[1]} {cube1[2]} {cube2[0]} {cube2[1]} {cube2[2]} {cube3[0]} {cube3[1]} {cube3[2]}")
                        found = True
                        break

if not found:
    print("No")
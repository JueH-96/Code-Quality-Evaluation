# YOUR CODE HERE
def main():
    V1, V2, V3 = map(int, input().split())
    
    # Total volume of one cube
    cube_volume = 7 * 7 * 7
    
    # Total volume covered by at least one cube
    total_volume = V1 + V2 + V3
    
    # Since we have three cubes, the total volume covered by at least one cube is:
    # 3 * cube_volume - 3 * V2 - 2 * V3
    # But this is not directly useful. Instead, we can use the inclusion-exclusion principle.
    
    # Let's consider the possible overlaps:
    # V1 = volume in exactly one cube
    # V2 = volume in exactly two cubes
    # V3 = volume in all three cubes
    
    # The total volume covered by at least one cube is V1 + V2 + V3
    # The total volume covered by exactly one cube is V1
    # The total volume covered by exactly two cubes is V2
    # The total volume covered by all three cubes is V3
    
    # We need to find positions of three cubes such that:
    # V1 = volume in exactly one cube
    # V2 = volume in exactly two cubes
    # V3 = volume in all three cubes
    
    # Let's try to find a configuration where the cubes overlap in a specific way.
    # For example, let's assume that the cubes overlap in such a way that:
    # - The intersection of all three cubes is a smaller cube of side length s, where s is such that s^3 = V3
    # - The pairwise intersections are such that the volume in exactly two cubes is V2
    # - The volume in exactly one cube is V1
    
    # Let's try to find s such that s^3 = V3
    s = round(V3 ** (1/3))
    if s ** 3 != V3:
        print("No")
        return
    
    # Now, we need to arrange the cubes such that the intersection of all three is a cube of side s
    # Let's assume that the cubes are aligned along the x, y, and z axes
    
    # Let's place the first cube at (0, 0, 0)
    a1, b1, c1 = 0, 0, 0
    
    # The second cube should overlap with the first cube in such a way that the intersection is a cube of side s
    # Let's place the second cube at (0, 0, 7 - s)
    a2, b2, c2 = 0, 0, 7 - s
    
    # The third cube should overlap with both the first and second cubes in such a way that the intersection is a cube of side s
    # Let's place the third cube at (0, 7 - s, 0)
    a3, b3, c3 = 0, 7 - s, 0
    
    # Now, let's calculate the volumes:
    # Volume in exactly one cube:
    # For each cube, the volume not overlapping with the others is cube_volume - (volume overlapping with one cube) - (volume overlapping with both cubes)
    # But this is complex. Instead, let's calculate the total volume covered by at least one cube:
    # total_volume = 3 * cube_volume - 3 * (volume of pairwise intersection) + (volume of triple intersection)
    # We know that the volume of triple intersection is V3 = s^3
    # The volume of pairwise intersection is (7 * 7 * s) for each pair, but we need to subtract the triple intersection
    # So, volume of pairwise intersection = 7 * 7 * s - s^3
    # Therefore, total_volume = 3 * cube_volume - 3 * (7 * 7 * s - s^3) + s^3
    # Simplifying:
    # total_volume = 3 * 343 - 3 * (49s - s^3) + s^3
    # total_volume = 1029 - 147s + 3s^3 + s^3
    # total_volume = 1029 - 147s + 4s^3
    
    # But we know that total_volume = V1 + V2 + V3
    # So, 1029 - 147s + 4s^3 = V1 + V2 + V3
    
    # Let's check if this holds:
    calculated_total = 1029 - 147 * s + 4 * s ** 3
    if calculated_total != V1 + V2 + V3:
        print("No")
        return
    
    # Now, let's calculate V1, V2, V3 based on the positions:
    # Volume in exactly one cube:
    # For cube 1: volume not overlapping with cube 2 or cube 3
    # For cube 2: volume not overlapping with cube 1 or cube 3
    # For cube 3: volume not overlapping with cube 1 or cube 2
    
    # Volume in exactly two cubes:
    # For each pair of cubes, the volume overlapping only those two
    
    # Volume in all three cubes: s^3
    
    # Let's compute V1, V2, V3:
    # Volume in exactly one cube:
    # For cube 1: (7 * 7 * 7) - (7 * 7 * s) - (7 * 7 * s) + (s * s * s)
    # Wait, this is not accurate. Instead, let's think differently.
    
    # The volume in exactly one cube is:
    # For cube 1: volume of cube 1 minus the volume overlapping with cube 2 and cube 3
    # So, V1_cube1 = cube_volume - (volume overlapping with cube 2) - (volume overlapping with cube 3) + (volume overlapping with both cube 2 and cube 3)
    # Similarly for cube 2 and cube 3
    
    # Volume overlapping with cube 2: 7 * 7 * s
    # Volume overlapping with cube 3: 7 * 7 * s
    # Volume overlapping with both cube 2 and cube 3: s^3
    
    # So, V1_cube1 = 343 - 49s - 49s + s^3 = 343 - 98s + s^3
    # Similarly for cube 2 and cube 3
    
    V1_cube1 = 343 - 98 * s + s ** 3
    V1_cube2 = 343 - 98 * s + s ** 3
    V1_cube3 = 343 - 98 * s + s ** 3
    
    V1_total = V1_cube1 + V1_cube2 + V1_cube3
    
    # Volume in exactly two cubes:
    # For each pair, the volume overlapping only those two
    # Volume overlapping cube 1 and cube 2: 7 * 7 * s - s^3
    # Similarly for cube 1 and cube 3, and cube 2 and cube 3
    
    V2_pair12 = 49 * s - s ** 3
    V2_pair13 = 49 * s - s ** 3
    V2_pair23 = 49 * s - s ** 3
    
    V2_total = V2_pair12 + V2_pair13 + V2_pair23
    
    # Volume in all three cubes: s^3
    V3_total = s ** 3
    
    if V1_total == V1 and V2_total == V2 and V3_total == V3:
        print("Yes")
        print(a1, b1, c1, a2, b2, c2, a3, b3, c3)
    else:
        print("No")

if __name__ == "__main__":
    main()
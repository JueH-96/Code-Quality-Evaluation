def main():
    import sys
    import math
    from itertools import permutations

    # Read the 3x3 grid as a flat list c[0..8]
    c = []
    for _ in range(3):
        c.extend(list(map(int, sys.stdin.readline().split())))

    # Define the 8 lines of interest (in 0-based index)
    lines = [
        (0, 1, 2),  # row 1
        (3, 4, 5),  # row 2
        (6, 7, 8),  # row 3
        (0, 3, 6),  # col 1
        (1, 4, 7),  # col 2
        (2, 5, 8),  # col 3
        (0, 4, 8),  # main diagonal
        (2, 4, 6),  # anti-diagonal
    ]

    # Total permutations of 9 squares
    total_permutations = math.factorial(9)
    valid_count = 0

    # Check every permutation of the 9 positions
    for perm in permutations(range(9)):
        # pos[x] = index of position x in the permutation
        pos = [0] * 9
        for i, square_idx in enumerate(perm):
            pos[square_idx] = i

        disappointed = False
        # Check each of the 8 lines
        for x, y, z in lines:
            # Sort squares x, y, z by the order in which they appear (pos)
            ordered = sorted([x, y, z], key=lambda idx: pos[idx])
            s1, s2, s3 = ordered
            # If first two seen have the same number but the last is different => disappointed
            if c[s1] == c[s2] != c[s3]:
                disappointed = True
                break

        if not disappointed:
            valid_count += 1

    # Print probability
    print(valid_count / total_permutations)

# Do not forget to call main().
if __name__ == "__main__":
    main()
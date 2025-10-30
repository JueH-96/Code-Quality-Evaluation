def main():
    import sys

    # Read input
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    grid = [data[2 + i] for i in range(N)]

    # Define positions that must be '#' and positions that must be '.'
    hash_positions = [(k, l) for k in range(3) for l in range(3)] + [(k + 6, l + 6) for k in range(3) for l in range(3)]
    dot_positions = [
        (0, 3), (1, 3), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3),
        (6, 5), (7, 5), (8, 5), (5, 6), (5, 7), (5, 8), (5, 5)
    ]

    # List to store valid (i, j) pairs
    results = []

    # Iterate over all possible 9x9 regions
    for i in range(N - 8):
        for j in range(M - 8):
            # Check all positions that must be '#'
            all_hashes = all(grid[i + k][j + l] == '#' for k, l in hash_positions)
            # Check all positions that must be '.'
            all_dots = all(grid[i + k][j + l] == '.' for k, l in dot_positions)
            # If all conditions are satisfied
            if all_hashes and all_dots:
                # 1-indexed coordinates
                results.append((i + 1, j + 1))

    # Sort the results
    results.sort()

    # Print the results
    for i, j in results:
        print(i, j)

if __name__ == "__main__":
    main()
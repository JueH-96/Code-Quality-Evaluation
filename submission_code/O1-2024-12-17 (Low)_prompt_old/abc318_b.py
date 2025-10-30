def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    coords = data[1:]

    # A direct approach is to use a 2D boolean array (101x101).
    # Then, for each rectangle, mark the cells it covers as True.

    covered = [[False] * 101 for _ in range(101)]
    idx = 0
    for _ in range(N):
        A = int(coords[idx]); B = int(coords[idx+1])
        C = int(coords[idx+2]); D = int(coords[idx+3])
        idx += 4
        # Mark covered area
        for x in range(A, B):
            for y in range(C, D):
                covered[x][y] = True

    # Calculate the total covered cells
    area = sum(sum(row) for row in covered)
    print(area)

# Call the solve() function
solve()
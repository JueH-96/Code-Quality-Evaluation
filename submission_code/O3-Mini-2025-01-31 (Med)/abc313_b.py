def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # Initialize a 2D boolean array for transitive closure: reach[i][j] is True if person i is stronger than j.
    reach = [[False] * N for _ in range(N)]
    
    # Set the known relations from input.
    for _ in range(M):
        a = int(next(it)) - 1
        b = int(next(it)) - 1
        reach[a][b] = True
    
    # Compute the transitive closure using Floyd-Warshall.
    for k in range(N):
        for i in range(N):
            if reach[i][k]:
                for j in range(N):
                    if reach[k][j]:
                        reach[i][j] = True

    # Identify if there's a candidate who is known to be stronger than every other person.
    candidate = -1
    for i in range(N):
        # A candidate must be stronger than every other person.
        if all(i == j or reach[i][j] for j in range(N)):
            candidate = i + 1
            break

    sys.stdout.write(str(candidate))


if __name__ == '__main__':
    main()
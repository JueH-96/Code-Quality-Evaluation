def main():
    import sys
    input = sys.stdin.readline

    # read input
    N, M = map(int, input().split())
    # We create a matrix "stronger" where stronger[i][j] = True will mean
    # that (person i+1) is known (or can be deduced) to be stronger than (person j+1).
    # Initialize matrix.
    stronger = [[False] * N for _ in range(N)]
    
    # Process the given M relations.
    for _ in range(M):
        a, b = map(int, input().split())
        stronger[a - 1][b - 1] = True

    # Use Floyd-Warshall algorithm to compute the transitive closure.
    # After this, stronger[i][j] == True if i is stronger than j through some chain.
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if stronger[i][k] and stronger[k][j]:
                    stronger[i][j] = True

    # We now want to check if a candidate exists such that it is deduced to be stronger than every other.
    candidate = -1
    count_candidates = 0
    for i in range(N):
        # Check that for every other j, i is stronger than j.
        valid = True
        for j in range(N):
            if i == j:
                continue
            if not stronger[i][j]:
                valid = False
                break
        if valid:
            candidate = i + 1
            count_candidates += 1

    # We print candidate if it is unique, otherwise print -1.
    if count_candidates == 1:
        sys.stdout.write(str(candidate))
    else:
        sys.stdout.write("-1")

if __name__ == '__main__':
    main()
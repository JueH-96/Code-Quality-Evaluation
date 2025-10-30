import sys

def count_black_squares(N, P, queries):
    """
    Count the number of black squares in each query area.

    Args:
    N (int): The size of the grid.
    P (list): The grid pattern.
    queries (list): The list of queries.

    Returns:
    list: The number of black squares in each query area.
    """
    results = []
    for A, B, C, D in queries:
        count = 0
        for i in range(A, C + 1):
            for j in range(B, D + 1):
                if P[i % N][j % N] == 'B':
                    count += 1
        results.append(count)
    return results

def main():
    # Read input from stdin
    N, Q = map(int, sys.stdin.readline().split())
    P = [list(sys.stdin.readline().strip()) for _ in range(N)]
    queries = [list(map(int, sys.stdin.readline().split())) for _ in range(Q)]

    # Count black squares for each query
    results = count_black_squares(N, P, queries)

    # Write results to stdout
    for result in results:
        sys.stdout.write(str(result) + '
')

if __name__ == '__main__':
    main()
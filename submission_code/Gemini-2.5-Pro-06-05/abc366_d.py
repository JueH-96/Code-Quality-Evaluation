import sys

def main():
    """
    Solves the 3D range sum query problem using prefix sums.
    """
    # Use a faster I/O method for handling large numbers of queries.
    input = sys.stdin.readline

    # Step 1: Read problem size N and the 3D grid A.
    try:
        N = int(input())
    except (IOError, ValueError):
        # Handle cases where input might be empty, e.g., in some online judges.
        return
        
    # A[x][y][z] will correspond to the problem's A_{x+1, y+1, z+1}.
    A = [[[0] * N for _ in range(N)] for _ in range(N)]
    for x in range(N):
        for y in range(N):
            line = list(map(int, input().split()))
            for z in range(N):
                A[x][y][z] = line[z]

    # Step 2: Precompute the 3D prefix sum table.
    # We use a table of size (N+1)x(N+1)x(N+1) to simplify 1-based indexing.
    prefix_sum = [[[0] * (N + 1) for _ in range(N + 1)] for _ in range(N + 1)]

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            for k in range(1, N + 1):
                # Get the value from the original 0-indexed array A.
                val = A[i - 1][j - 1][k - 1]
                
                # Apply the inclusion-exclusion principle to compute the prefix sum.
                prefix_sum[i][j][k] = (
                    val
                    + prefix_sum[i - 1][j][k]
                    + prefix_sum[i][j - 1][k]
                    + prefix_sum[i][j][k - 1]
                    - prefix_sum[i - 1][j - 1][k]
                    - prefix_sum[i - 1][j][k - 1]
                    - prefix_sum[i][j - 1][k - 1]
                    + prefix_sum[i - 1][j - 1][k - 1]
                )

    # Step 3: Read and process Q queries.
    try:
        Q = int(input())
    except (IOError, ValueError):
        return

    results = []
    for _ in range(Q):
        Lx, Rx, Ly, Ry, Lz, Rz = map(int, input().split())

        # Calculate the sum of the query sub-cuboid in O(1) time
        # using the precomputed table and the inclusion-exclusion principle.
        ans = (
            prefix_sum[Rx][Ry][Rz]
            - prefix_sum[Lx - 1][Ry][Rz]
            - prefix_sum[Rx][Ly - 1][Rz]
            - prefix_sum[Rx][Ry][Lz - 1]
            + prefix_sum[Lx - 1][Ly - 1][Rz]
            + prefix_sum[Lx - 1][Ry][Lz - 1]
            + prefix_sum[Rx][Ly - 1][Lz - 1]
            - prefix_sum[Lx - 1][Ly - 1][Lz - 1]
        )
        results.append(str(ans))

    # Step 4: Print all results.
    print("
".join(results))

if __name__ == "__main__":
    main()
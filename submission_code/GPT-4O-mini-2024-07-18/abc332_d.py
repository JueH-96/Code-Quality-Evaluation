def min_operations_to_identical_grids(H, W, A, B):
    from collections import defaultdict

    # Check if it's possible to make A identical to B
    def can_transform(A, B):
        # Count occurrences of each row in A and B
        row_count_A = defaultdict(int)
        row_count_B = defaultdict(int)

        for row in A:
            row_count_A[tuple(row)] += 1
        for row in B:
            row_count_B[tuple(row)] += 1

        return row_count_A == row_count_B

    if not can_transform(A, B):
        return -1

    # Calculate the minimum number of operations
    def count_operations(A, B):
        # Create a mapping of rows in B to their indices
        row_index_B = {tuple(row): i for i, row in enumerate(B)}
        target_rows = [row_index_B[tuple(row)] for row in A]

        # Count the number of swaps needed to sort target_rows
        def count_swaps(arr):
            visited = [False] * len(arr)
            swaps = 0

            for i in range(len(arr)):
                if visited[i] or arr[i] == i:
                    continue

                cycle_size = 0
                j = i
                while not visited[j]:
                    visited[j] = True
                    j = arr[j]
                    cycle_size += 1

                if cycle_size > 0:
                    swaps += (cycle_size - 1)

            return swaps

        return count_swaps(target_rows)

    row_operations = count_operations(A, B)

    # Now we need to handle columns
    # Transpose A and B to handle columns as rows
    A_transposed = list(zip(*A))
    B_transposed = list(zip(*B))

    # Check if it's possible to make A_transposed identical to B_transposed
    if not can_transform(A_transposed, B_transposed):
        return -1

    column_operations = count_operations(A_transposed, B_transposed)

    return row_operations + column_operations

import sys
input = sys.stdin.read
data = input().splitlines()

H, W = map(int, data[0].split())
A = [list(map(int, data[i + 1].split())) for i in range(H)]
B = [list(map(int, data[i + 1 + H].split())) for i in range(H)]

result = min_operations_to_identical_grids(H, W, A, B)
print(result)
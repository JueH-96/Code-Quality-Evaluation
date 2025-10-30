import sys

def solve():
    # Read N and W
    N, W = map(int, sys.stdin.readline().split())

    # Store initial blocks and group by column
    initial_blocks = [] # 0-indexed list for potential future use (though not strictly needed for this solution)
    blocks_by_column = {x: [] for x in range(1, W + 1)}
    for i in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        initial_blocks.append((X, Y))
        blocks_by_column[X].append((Y, i + 1)) # Store (Y, original_index)

    # Sort blocks in each column by Y coordinate
    for x in range(1, W + 1):
        blocks_by_column[x].sort()

    # Check if any column is initially empty
    has_empty_column = False
    for x in range(1, W + 1):
        if not blocks_by_column[x]:
            has_empty_column = True
            break

    # If any column is empty, row 1 can never be full. No blocks removed.
    # Answer is always Yes for initial blocks.
    if has_empty_column:
        Q = int(sys.stdin.readline())
        for _ in range(Q):
            # T, A = map(int, sys.stdin.readline().split()) # No need to read T, A if always Yes
            # We still need to consume the input line for the query
            sys.stdin.readline()
            print("Yes")
        return

    # If all columns have blocks initially, calculate K_min and all_lowest_at_one flag
    K_min = min(len(blocks_by_column[x]) for x in range(1, W + 1))

    all_lowest_at_one = True
    for x in range(1, W + 1):
        # blocks_by_column[x][0][0] is the minimum initial Y in column x
        if blocks_by_column[x][0][0] != 1:
            all_lowest_at_one = False
            break

    # Build block_ranks {original_index: rank}
    block_ranks = {}
    for x in range(1, W + 1):
        for rank, (y, idx) in enumerate(blocks_by_column[x]):
            block_ranks[idx] = rank + 1 # 1-based rank

    # Process queries
    Q = int(sys.stdin.readline())
    for _ in range(Q):
        T, A = map(int, sys.stdin.readline().split()) # A is original_index (1 to N)

        # Get the rank of block A in its column
        R_A = block_ranks[A]

        # Determine if the block is removed by time T
        is_removed_by_T = False
        if all_lowest_at_one:
            # Block B_{X, R} is removed at time R if R <= K_min.
            # It is removed by time T if R <= K_min AND R <= T.
            if R_A <= K_min and R_A <= T:
                is_removed_by_T = True
        else:
            # Block B_{X, R} is removed at time R + 1 if R <= K_min.
            # It is removed by time T if R <= K_min AND R + 1 <= T.
            if R_A <= K_min and R_A + 1 <= T:
                is_removed_by_T = True

        # Block exists at T+0.5 if it is NOT removed by time T
        if is_removed_by_T:
            print("No")
        else:
            print("Yes")

solve()
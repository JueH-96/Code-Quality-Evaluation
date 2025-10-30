import sys
from collections import defaultdict
import bisect

# Increase recursion depth - might not be needed but good practice for competitive programming
# sys.setrecursionlimit(3000)

def solve():
    # Read N and W
    N, W = map(int, sys.stdin.readline().split())
    
    # Store initial block positions and group by column
    blocks_by_col = defaultdict(list)
    # initial_pos is not directly needed for the final logic, but could be useful for debugging
    # initial_pos = {} 

    for i in range(1, N + 1):
        x, y = map(int, sys.stdin.readline().split())
        blocks_by_col[x].append((y, i))
        # initial_pos[i] = (x, y)

    # Sort blocks within each column by initial height
    # And create the map from original block index to (column, initial_rank)
    k = {} # Number of initial blocks in each column (k_x for column x)
    orig_idx_to_col_rank = {}
    
    # Process columns 1 to W
    for x in range(1, W + 1):
        blocks_by_col[x].sort()
        k[x] = len(blocks_by_col[x])
        for rank, (y, orig_idx) in enumerate(blocks_by_col[x]):
            orig_idx_to_col_rank[orig_idx] = (x, rank + 1) # rank is 1-based

    # Calculate min_k_positive: minimum number of blocks in any column that is initially non-empty.
    # If all columns are initially non-empty, this is min(k_w).
    # If any column is initially empty, row 1 can never be full, so this value represents infinity.
    min_k_positive = float('inf')
    all_k_positive = True

    for x in range(1, W + 1):
        if k[x] == 0:
            all_k_positive = False
            break # If even one column is empty, min_k_positive remains inf
        else:
            min_k_positive = min(min_k_positive, k[x])

    # If all_k_positive is False, min_k_positive should remain float('inf')
    # If all_k_positive is True, min_k_positive holds min(k_w)

    # Read number of queries
    Q = int(sys.stdin.readline())

    # Process queries
    for _ in range(Q):
        Tj, Aj = map(int, sys.stdin.readline().split())

        # Get the original column and initial rank of block Aj
        # We are guaranteed that Aj is a valid block index from 1 to N
        # Thus, original_pos[Aj] and orig_idx_to_col_rank[Aj] must exist
        x, initial_rank = orig_idx_to_col_rank[Aj]

        # Calculate the number of blocks removed from column x by the end of time step Tj.
        # This is equal to the number of times row 1 became full during steps 1 to Tj,
        # because each time row 1 becomes full, the lowest block in every non-empty column
        # (which is at height 1 due to sequential falling) is removed. This process reduces
        # the number of blocks in a column by 1, as long as the column is not yet exhausted.
        # The number of times row 1 becomes full in steps 1..Tj depends on how many
        # steps can pass before at least one column is completely emptied by row 1 clearing.
        # If any column starts empty (min_k_positive = inf), row 1 is never full.
        # If all columns start non-empty (min_k_positive is finite), row 1 is full
        # at steps 1, 2, ..., min_k_positive. After min_k_positive steps, the column
        # with the minimum initial number of blocks becomes empty, and row 1 cannot be full anymore.
        # The number of times row 1 is full during steps 1..Tj is min(Tj, min_k_positive)
        # if min_k_positive is finite, and 0 otherwise.

        times_row1_full_upto_Tj = 0
        if min_k_positive != float('inf'):
             times_row1_full_upto_Tj = min(Tj, min_k_positive)

        # The number of blocks removed from column x by the end of step Tj is
        # equal to the total number of times row 1 was full up to Tj, because
        # column x (since k[x] >= 1 for the queried block) contributes a block
        # to be removed every time row 1 is full, until column x itself is exhausted.
        # However, the condition for row 1 being full depends on the minimum capacity
        # across *all* columns. Once the bottleneck column (with min_k_positive blocks)
        # is exhausted after min_k_positive row-clearing events, row 1 stops becoming full.
        # Thus, column x participates in row clearing exactly times_row1_full_upto_Tj times.

        removed_count_from_col_x = times_row1_full_upto_Tj

        # Block Aj (with initial_rank in column x) exists at Tj+0.5 if its initial rank
        # is strictly greater than the number of blocks removed from its column.
        # The blocks removed from column x are those with the lowest initial ranks:
        # rank 1, rank 2, ..., rank removed_count_from_col_x.
        # So block with initial rank 'initial_rank' exists iff initial_rank is not in
        # the set {1, 2, ..., removed_count_from_col_x}.
        # This is true iff initial_rank > removed_count_from_col_x.
        if initial_rank > removed_count_from_col_x:
            print("Yes")
        else:
            print("No")

solve()
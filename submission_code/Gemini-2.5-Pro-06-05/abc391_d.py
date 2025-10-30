import sys

def main():
    """
    Main function to read input, solve the problem, and print output.
    """
    
    # Use fast I/O, a common practice in competitive programming.
    input = sys.stdin.readline

    try:
        N, W = map(int, input().split())

        # Group blocks by column.
        # cols[x] stores tuples of (Y, original_index) for column x.
        # Using a list of lists with size W+1 for 1-based indexing.
        cols = [[] for _ in range(W + 1)]
        for i in range(1, N + 1):
            X, Y = map(int, input().split())
            cols[X].append((Y, i))
            
        # Sort blocks within each column by their initial Y-coordinate to determine rank.
        for x in range(1, W + 1):
            cols[x].sort()
            
        # Store rank and column for each block for O(1) lookup.
        # block_info[original_index] = (column_number, rank_in_column)
        block_info = [None] * (N + 1)
        initial_counts = [0] * (W + 1)
        for x in range(1, W + 1):
            count = len(cols[x])
            initial_counts[x] = count
            for rank, (Y, original_index) in enumerate(cols[x], 1):
                block_info[original_index] = (x, rank)
                
        # Determine `min_count`, the number of blocks in the sparsest column.
        # This is the total number of full row removals that can ever occur.
        min_count = N + 1 
        has_empty_col = False
        if W > N or any(count == 0 for count in initial_counts[1:]):
             has_empty_col = True
        
        if has_empty_col:
            min_count = 0
        else:
            min_count = min(initial_counts[1:])

        # If `min_count` is 0, no rows are ever removed. All blocks exist forever.
        if min_count == 0:
            Q = int(input())
            for _ in range(Q):
                input() # Read and discard query
                sys.stdout.write("Yes
")
            return

        # Calculate `t1`, the time of the first row removal.
        first_ys = [cols[x][0][0] for x in range(1, W + 1)]
        t1 = max(first_ys)
        
        Q = int(input())
        
        # Process each query
        for _ in range(Q):
            T, A = map(int, input().split())
            
            x, k = block_info[A]
            
            # If the block's rank `k` is greater than the total number of possible
            # removals `min_count`, it will never be removed.
            if k > min_count:
                sys.stdout.write("Yes
")
                continue
                
            # Otherwise, the block is destined to be removed.
            # The k-th removal event happens at time `t1 + k - 1`.
            t_remove = t1 + k - 1
            
            # The block exists at time T+0.5 if time T is before its removal time.
            if T >= t_remove:
                sys.stdout.write("No
")
            else:
                sys.stdout.write("Yes
")
                
    except (IOError, IndexError):
        # Graceful exit on I/O or index errors, though not expected with valid input.
        return

if __name__ == "__main__":
    main()
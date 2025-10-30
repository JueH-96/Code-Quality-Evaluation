import sys

def main():
    """
    Main function to read input, solve the problem, and print the output.
    """
    
    # Using sys.stdin.readline for faster I/O in competitive programming
    input = sys.stdin.readline
    
    # Read problem parameters
    N = int(input())
    S = input().strip()
    Q = int(input())

    # Store all queries to process them after finding the last global operation
    queries = []
    for _ in range(Q):
        line = input().split()
        t = int(line[0])
        x = int(line[1])
        c = line[2]
        queries.append((t, x, c))

    # --- Algorithm Core ---
    
    # Find the index and type of the last global case-change operation (type 2 or 3).
    # This is the pivotal operation that determines the base case of most characters.
    last_global_op_idx = -1
    last_global_op_type = -1

    # Scan queries in reverse to find the last one of type 2 or 3.
    for i in range(Q - 1, -1, -1):
        if queries[i][0] > 1:
            last_global_op_idx = i
            last_global_op_type = queries[i][0]
            break

    # Convert the string to a list for mutable character operations
    s_list = list(S)
    
    # If no global operation was found, all operations are type 1.
    # We can process them all sequentially.
    if last_global_op_idx == -1:
        for t, x, c in queries:
            # All must be type 1, but we check for robustness.
            if t == 1:
                s_list[x - 1] = c
    else:
        # If there is a last global operation, we process operations in three stages.
        
        # Stage 1: Apply type 1 changes that occur *before* the last global operation.
        for i in range(last_global_op_idx):
            t, x, c = queries[i]
            if t == 1:
                s_list[x - 1] = c
        
        # Stage 2: Apply the effect of the last global operation itself.
        # This is the only O(N) case-changing operation performed.
        if last_global_op_type == 2:
            s_list = [char.lower() for char in s_list]
        else: # type == 3
            s_list = [char.upper() for char in s_list]
            
        # Stage 3: Apply type 1 changes that occur *after* the last global operation.
        # These changes are not affected by the global case change.
        for i in range(last_global_op_idx + 1, Q):
            t, x, c = queries[i]
            if t == 1:
                s_list[x - 1] = c

    # Join the list back into a string and print the final result.
    print("".join(s_list))

if __name__ == "__main__":
    main()
import sys

def solve():
    # Read N, the number of garbage types
    n = int(sys.stdin.readline())

    # Store garbage collection parameters (q_i, r_i) for each type
    # Use a list where index i stores parameters for garbage type i+1 (1-based)
    garbage_info = []
    for _ in range(n):
        q, r = map(int, sys.stdin.readline().split())
        # Append tuple (q, r) to the list
        garbage_info.append((q, r))

    # Read Q, the number of queries
    q_queries = int(sys.stdin.readline())

    # Prepare a list to store the answers for each query
    answers = []

    # Process each query
    for _ in range(q_queries):
        # Read query: garbage type t (1-based) and day d it's put out
        t, d = map(int, sys.stdin.readline().split())
        
        # Adjust type t from 1-based to 0-based index to access the list
        t_idx = t - 1
        
        # Retrieve the collection frequency q and remainder r for this type
        q, r = garbage_info[t_idx]
        
        # Calculate the next collection day `ans` such that ans >= d and ans % q == r
        
        # Find the remainder of day d when divided by the frequency q
        # This tells us where day d falls within a cycle of length q
        d_rem = d % q
        
        # Determine the next collection day based on the remainder relative to r
        
        # Case 1: If d itself is a collection day (d % q == r)
        # The garbage is collected on day d.
        if d_rem == r:
            ans = d
            
        # Case 2: If d is before the collection day r within the current cycle [k*q, (k+1)*q)
        # where k = d // q. The collection day in this cycle is k*q + r.
        # For example, if q=7, r=3, and d=1, then d_rem=1. Since 1 < 3, the collection
        # day in the current cycle (0*7 + 3 = 3) is after day 1.
        elif d_rem < r:
            # The next collection day is k*q + r.
            # Since d = k*q + d_rem and d_rem < r, we have k*q + r > d.
            # This is the smallest collection day that is greater than or equal to d.
            # We can calculate k*q as d - d_rem. So ans = (d - d_rem) + r.
            # Alternatively, using integer division: ans = (d // q) * q + r
            ans = (d // q) * q + r
            
        # Case 3: If d is after the collection day r within the current cycle
        # For example, if q=7, r=3, and d=4, then d_rem=4. Since 4 > 3, the collection
        # day in the current cycle (0*7 + 3 = 3) has already passed (is less than d).
        else: # d_rem > r
            # The collection day in the current cycle (k*q + r) is already passed (< d).
            # The next collection day must be in the following cycle: (k+1)*q + r.
            # This day is guaranteed to be > d.
            ans = (d // q + 1) * q + r
            
        # Add the calculated answer to the list
        # Convert to string for efficient joining later
        answers.append(str(ans))

    # Print all collected answers, each on a new line
    # Using '
'.join is generally faster than printing in a loop for large number of lines
    print("
".join(answers))

# Execute the solve function to run the program
solve()
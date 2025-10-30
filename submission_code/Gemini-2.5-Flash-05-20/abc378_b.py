# YOUR CODE HERE
import sys

def solve():
    # Read N, the number of garbage types
    N = int(sys.stdin.readline())

    # Store the collection rules (q_i, r_i).
    # garbage_rules[i] will store (q_{i+1}, r_{i+1}) due to 0-based indexing.
    garbage_rules = []
    for _ in range(N):
        q, r = map(int, sys.stdin.readline().split())
        garbage_rules.append((q, r))

    # Read Q, the number of queries
    Q = int(sys.stdin.readline())

    # Process each query
    for _ in range(Q):
        # Read the garbage type t_j and the day d_j it was put out.
        t_j, d_j = map(int, sys.stdin.readline().split())
        
        # Get the collection parameters (q, r) for the specified garbage type.
        # t_j is 1-indexed in the problem description, so we use t_j - 1 for 0-based list access.
        q, r = garbage_rules[t_j - 1]

        # Calculate the remainder of d_j when divided by q.
        # This tells us where d_j falls within the collection cycle defined by q.
        current_remainder = d_j % q

        # Determine how many days to add to d_j to reach the next collection day.
        # The goal is to find the smallest non-negative 'x' such that (d_j + x) % q == r.
        # This is equivalent to finding 'x' such that (current_remainder + x) % q == r.
        #
        # The formula (r - current_remainder + q) % q calculates the non-negative difference
        # needed to reach 'r' from 'current_remainder' in a modulo 'q' system.
        # - If current_remainder == r: (r - r + q) % q = q % q = 0. So, 0 days are added.
        #   The garbage is collected on day d_j itself.
        # - If current_remainder < r: (r - current_remainder) is positive. Adding 'q' and
        #   taking modulo 'q' will yield (r - current_remainder) itself.
        # - If current_remainder > r: (r - current_remainder) is negative. Adding 'q' effectively
        #   wraps it around to the correct positive difference within the cycle.
        days_to_add = (r - current_remainder + q) % q
        
        # The next collection day is the day the garbage was put out plus the calculated days to add.
        next_collection_day = d_j + days_to_add
        
        # Print the result for the current query.
        print(next_collection_day)

# Call the solve function to run the program
solve()
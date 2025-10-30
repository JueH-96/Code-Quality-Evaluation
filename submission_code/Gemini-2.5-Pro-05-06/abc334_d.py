# YOUR CODE HERE
import sys
import bisect

# Main logic
def solve():
    # Use sys.stdin.readline for faster input in competitive programming
    input_readline = sys.stdin.readline

    N, Q = map(int, input_readline().split())
    R_values = list(map(int, input_readline().split()))

    # To maximize the number of sleighs, we should always pick sleighs
    # that require the fewest reindeer. So, sort R_values.
    R_values.sort()

    # Calculate prefix sums.
    # prefix_sums[i] will store the total reindeer cost to pull the (i+1) cheapest sleighs.
    # For example, prefix_sums[0] is cost for 1 sleigh (the R_values[0]).
    # prefix_sums[N-1] is cost for all N sleighs (sum of all R_values).
    # Constraints: 1 <= N. So N is at least 1.
    prefix_sums = [0] * N 
    
    current_sum = 0
    for i in range(N):
        current_sum += R_values[i]
        prefix_sums[i] = current_sum
    
    # Store results for all queries to print them at once at the end.
    # This is often faster than printing line by line due to I/O buffering.
    results_buffer = []
    for _ in range(Q):
        X = int(input_readline()) # Number of available reindeer for this query
        
        # We need to find the maximum k such that the sum of the k smallest R_values is <= X.
        # This is equivalent to finding the largest k such that prefix_sums[k-1] <= X (for k>=1).
        # bisect_right(a, val) returns an index `idx` such that all elements a[0...idx-1] are <= val,
        # and all elements a[idx...end] are > val.
        # So, `idx` is the count of elements in `a` that are less than or equal to `val`.
        # In our `prefix_sums` array, an element `prefix_sums[j]` represents the cost for `j+1` sleighs.
        # If `idx` elements of `prefix_sums` are <= X, it means we can afford
        # the configurations involving 1 sleigh, 2 sleighs, ..., up to `idx` sleighs.
        # Thus, the maximum number of sleighs we can pull is `idx`.
        num_sleighs_can_pull = bisect.bisect_right(prefix_sums, X)
        results_buffer.append(str(num_sleighs_can_pull))
        
    # Print all results, each on a new line.
    # Adding a final newline character is standard.
    sys.stdout.write("
".join(results_buffer) + "
")

# Call the main solving function
solve()
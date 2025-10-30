# YOUR CODE HERE
import sys
import bisect

# Function to read space-separated integers from a line
def read_ints():
    """Reads a line from stdin, splits it by spaces, and maps each part to an integer."""
    return map(int, sys.stdin.readline().split())

# Main logic function
def solve():
    """Reads input, solves the problem, and prints the output."""
    # Read the number of villages
    n = int(sys.stdin.readline())
    
    # Read the coordinates of the villages into a list
    # Coordinates X are guaranteed to be sorted: X_1 < X_2 < ... < X_N
    x = list(read_ints())
    
    # Read the population of the villages into a list
    p = list(read_ints())
    
    # Read the number of queries
    q = int(sys.stdin.readline())

    # Calculate prefix sums for the populations P.
    # prefix_p[i] will store the sum of populations from village 0 to village i-1.
    # The size is n + 1 to handle sums over ranges conveniently. prefix_p[0] is initialized to 0.
    prefix_p = [0] * (n + 1)
    current_sum = 0
    for i in range(n):
        # Accumulate the population sum
        current_sum += p[i]
        # Store the cumulative sum up to index i (inclusive of p[i]) at prefix_p[i+1]
        # So, prefix_p[k] = P[0] + P[1] + ... + P[k-1]
        prefix_p[i+1] = current_sum

    # List to store the results for each query
    results = []

    # Process each query
    for _ in range(q):
        # Read the query range [L, R] (inclusive)
        l, r = read_ints()

        # Find the index of the first village whose coordinate X[k] is greater than or equal to L.
        # bisect_left(a, x) returns an insertion point which comes before (to the left of) any existing entries of x in a, 
        # and partitions a into two halves so that all(val < x for val in a[lo : i]) for the left half and 
        # all(val >= x for val in a[i : hi]) for the right half.
        # This gives the index of the first village included in the range [L, R].
        start_idx = bisect.bisect_left(x, l)

        # Find the index `j` such that all village coordinates X[k] with k < j satisfy X[k] <= R.
        # bisect_right(a, x) returns an insertion point which comes after (to the right of) any existing entries of x in a, 
        # and partitions a into two halves so that all(val <= x for val in a[lo : i]) for the left half and 
        # all(val > x for val in a[i : hi]) for the right half.
        # This gives the index immediately after the last village included in the range [L, R].
        end_idx = bisect.bisect_right(x, r)

        # The villages whose coordinates are within the range [L, R] correspond to
        # indices k such that `start_idx <= k <= end_idx - 1`.
        # The sum of populations for these villages is P[start_idx] + ... + P[end_idx - 1].
        # This sum can be efficiently calculated using the prefix sum array:
        # Sum = (P[0] + ... + P[end_idx - 1]) - (P[0] + ... + P[start_idx - 1])
        # Sum = prefix_p[end_idx] - prefix_p[start_idx]
        query_sum = prefix_p[end_idx] - prefix_p[start_idx]
        
        # Append the result of the current query to the results list
        results.append(query_sum)

    # Print all collected results, each on a new line.
    # Using '
'.join(map(str, results)) is generally more efficient for large numbers
    # of outputs compared to printing inside the loop using print() repeatedly.
    print('
'.join(map(str, results)))

# Execute the main solution function
solve()
# YOUR CODE HERE
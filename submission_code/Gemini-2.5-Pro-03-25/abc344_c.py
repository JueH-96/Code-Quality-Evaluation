# YOUR CODE HERE
import sys

# Read input function optimized for multiple integers on a line
def read_ints():
    """Reads a line of space-separated integers from stdin and returns a map object."""
    return map(int, sys.stdin.readline().split())

# Read single integer function
def read_int():
    """Reads a single integer from stdin."""
    return int(sys.stdin.readline())

def solve():
    """Solves the main problem based on the input."""
    # Read input sequences A, B, C
    n = read_int()
    a = list(read_ints())
    m = read_int()
    b = list(read_ints())
    l = read_int()
    c = list(read_ints())
    
    # Read queries
    q = read_int()
    x = list(read_ints())

    # Precompute all possible sums of one element from A and one element from B.
    # Store these sums in a set for efficient lookup (average O(1) time complexity).
    # The time complexity of this precomputation step is O(N * M).
    # Since N, M <= 100, N * M <= 10000, which is efficient.
    sum_ab = set()
    for val_a in a:
        for val_b in b:
            sum_ab.add(val_a + val_b)

    # Process each query X_i.
    # For each query, we need to check if there exists an element 'val_c' in C
    # such that X_i - val_c is present in the precomputed set `sum_ab`.
    # The time complexity of processing all queries is O(Q * L * SetLookupTime).
    # Assuming average O(1) set lookup time, this is O(Q * L).
    # Since Q <= 2*10^5 and L <= 100, Q * L <= 2*10^7, which is efficient.
    
    output_lines = [] # Use a list to store output lines for printing at the end
    for x_i in x:
        found = False # Flag to indicate if a solution (a+b+c = x_i) is found for the current query x_i
        
        # Iterate through each element 'val_c' in sequence C
        for val_c in c:
            # Calculate the required sum 'target' from sequences A and B
            # target = a + b = x_i - c
            target = x_i - val_c
            
            # Check if this 'target' sum exists in our precomputed set `sum_ab`
            if target in sum_ab:
                # If it exists, we found a combination (a, b, c) such that a + b + c = x_i
                found = True 
                break # Found a solution, no need to check other elements in C for this x_i
        
        # Append the result ("Yes" or "No") for the current query x_i to the output list
        if found:
            output_lines.append("Yes")
        else:
            output_lines.append("No")

    # Print all the results, each on a new line.
    # Using '
'.join() is generally efficient for printing multiple lines from a list of strings.
    print('
'.join(output_lines))

# Execute the main logic function
solve()
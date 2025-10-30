import sys

def solve():
    """
    This function reads the input, solves the problem, and prints the output.
    """
    try:
        # Using sys.stdin.readline() is generally faster for large inputs in competitive programming
        N_str = sys.stdin.readline()
        if not N_str:
            return
        N = int(N_str)
        
        A_str = sys.stdin.readline()
        if not A_str:
            return
        A = list(map(int, A_str.split()))
    except (IOError, ValueError):
        # Graceful exit on malformed input, though problem constraints usually guarantee valid input.
        return

    # A dictionary to store the most recent index of each number encountered.
    last_seen_index = {}
    
    # Initialize min_length to a value larger than any possible length.
    # float('inf') is a convenient way to represent infinity.
    min_length = float('inf')

    # Iterate through the array using enumerate to get both index and value.
    for i, value in enumerate(A):
        # Check if we have seen this value before.
        if value in last_seen_index:
            # If so, a repeat is found. The subarray is defined by the two
            # occurrences of this value.
            previous_index = last_seen_index[value]
            
            # The length of this subarray is the distance between indices plus one.
            length = i - previous_index + 1
            
            # We are looking for the shortest such subarray, so we update the minimum length.
            min_length = min(min_length, length)
        
        # After checking, update the last seen index for the current value.
        # This is crucial. By always updating, we ensure that for the next
        # time this value is encountered, we are comparing it with its
        # immediately preceding occurrence. This is sufficient to find the
        # overall shortest subarray with a repeat.
        last_seen_index[value] = i

    # After iterating through the entire array:
    # If min_length was never updated from its initial infinite value,
    # it means no repeated values were found in any subarray.
    if min_length == float('inf'):
        print(-1)
    else:
        # Otherwise, print the minimum length found.
        # The result might be a float, so we cast to int before printing.
        print(int(min_length))

# Run the solution
solve()
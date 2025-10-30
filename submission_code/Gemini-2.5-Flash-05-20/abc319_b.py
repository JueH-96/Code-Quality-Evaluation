import sys

def solve():
    N = int(sys.stdin.readline())

    # Initialize a list to store characters of the result string.
    # Pre-allocating the list of correct size can be slightly more efficient
    # than repeated appends for very large N, though for N=1000 it's not critical.
    result_chars = [''] * (N + 1)

    # Iterate through each index i from 0 to N to determine s_i
    for i in range(N + 1):
        # Initialize smallest_j_found to a value greater than any possible j (1-9).
        # This will indicate if no suitable j was found.
        smallest_j_found = 10 
        
        # Iterate through possible values of j from 1 to 9
        for j in range(1, 10): 
            # Condition 1: j must be a divisor of N
            if N % j == 0:
                # Calculate N/j. Since N % j == 0, N/j will be an integer.
                # Use integer division (//) to be explicit.
                threshold = N // j 
                
                # Condition 2: i must be a multiple of N/j (which is 'threshold')
                if i % threshold == 0:
                    # If both conditions are met, j is a candidate.
                    # Update smallest_j_found if the current j is smaller.
                    smallest_j_found = min(smallest_j_found, j)
        
        # After checking all j from 1 to 9 for the current i:
        if smallest_j_found == 10: 
            # If smallest_j_found is still 10, it means no suitable j was found.
            result_chars[i] = '-'
        else:
            # Otherwise, convert the smallest found j to its string representation.
            result_chars[i] = str(smallest_j_found)
            
    # Join the list of characters to form the final string and print it.
    print("".join(result_chars))

# This block ensures that solve() is called only when the script is executed directly.
if __name__ == '__main__':
    solve()
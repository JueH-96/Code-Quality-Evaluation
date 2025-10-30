import sys

def solve():
    # Read N from the first line of input.
    N = int(sys.stdin.readline())
    
    # Read the sequence A from the second line of input.
    A = list(map(int, sys.stdin.readline().split()))

    # This dictionary will store the last seen 1-based index for each number.
    # Key: a number from sequence A.
    # Value: the 1-based index of its most recent occurrence encountered so far.
    last_seen_pos = {}
    
    # This list will store the results B_1, B_2, ..., B_N.
    # Pre-allocating the list with a known size.
    B = [0] * N 

    # Iterate through the input array A using 0-based indexing.
    for i in range(N):
        current_val = A[i]
        # The problem uses 1-based indexing for positions.
        current_pos_1_based = i + 1

        # Retrieve the most recent 1-based position where current_val was seen.
        # If current_val is not in last_seen_pos (i.e., it's seen for the first time),
        # .get() returns the default value -1.
        previous_occurrence_pos = last_seen_pos.get(current_val, -1)
        
        # Store this result. B[i] corresponds to B_{i+1} in 1-based problem terms.
        B[i] = previous_occurrence_pos
        
        # Update the dictionary: current_val has now been seen at current_pos_1_based.
        last_seen_pos[current_val] = current_pos_1_based
        
    # Print the elements of B, separated by spaces, followed by a newline.
    sys.stdout.write(" ".join(map(str, B)) + "
")

if __name__ == '__main__':
    solve()
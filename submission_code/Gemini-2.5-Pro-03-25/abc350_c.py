# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read the size of the permutation
    N = int(sys.stdin.readline())
    
    # Read the permutation A. Store it in a 0-based indexed list.
    # A[i] will store the value A_{i+1} from the problem statement.
    A = list(map(int, sys.stdin.readline().split())) 

    # Create a position map 'pos'. 
    # pos[v] will store the 0-based index 'i' such that A[i] = v.
    # The size is N+1 because the values in the permutation are from 1 to N.
    # Index 0 of pos array is unused or could be considered 0.
    pos = [0] * (N + 1) 
    for i in range(N):
        pos[A[i]] = i # Initialize the position map based on the initial permutation A

    swaps = [] # List to store the sequence of swap operations performed. Each element is a tuple (i, j) with 1-based indices.

    # Iterate through the target positions. For each k from 1 to N,
    # we want to ensure that the value k is at index k-1 in the list A.
    for k in range(1, N + 1):
        # k_idx is the target 0-based index for the value k.
        k_idx = k - 1 
        
        # Check if the value currently at the target index A[k_idx] is already k.
        if A[k_idx] != k:
            # If A[k_idx] is not k, it means the value k is somewhere else.
            # We need to find where k is and swap it into the correct position k_idx.
            
            # Find the current 0-based index of value k using the position map.
            p_idx = pos[k] # This is the index where value k currently resides.
            
            # Get the value that is currently misplaced at the target index k_idx.
            val_at_k_idx = A[k_idx]
            
            # Perform the swap: Exchange the elements at indices k_idx and p_idx in the list A.
            # This moves value k to its correct position k_idx.
            # The value 'val_at_k_idx' moves from k_idx to p_idx.
            A[k_idx], A[p_idx] = A[p_idx], A[k_idx]
            
            # Record the swap operation. The problem requires 1-based indices.
            # Convert 0-based indices k_idx and p_idx to 1-based indices idx1 and idx2.
            idx1 = k_idx + 1
            idx2 = p_idx + 1
            # The problem requires the swap pair (i, j) to satisfy i < j.
            # Ensure this by using min and max.
            swaps.append((min(idx1, idx2), max(idx1, idx2)))
            
            # After the swap, the positions of two values have changed. Update the position map 'pos'.
            # The value k is now located at index k_idx.
            pos[k] = k_idx
            # The value 'val_at_k_idx' (which was originally at A[k_idx]) is now located at index p_idx.
            pos[val_at_k_idx] = p_idx

    # After the loop finishes, the list A should be sorted as [1, 2, ..., N].
    # Print the total number of swaps performed.
    print(len(swaps))
    
    # Print each swap operation (pair of 1-based indices) on a new line.
    for swap_pair in swaps:
        # Use f-string for concise formatted output.
        print(f"{swap_pair[0]} {swap_pair[1]}")

# Call the solve function to execute the logic when the script runs.
solve()
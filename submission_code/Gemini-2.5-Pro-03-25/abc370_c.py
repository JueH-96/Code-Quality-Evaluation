# YOUR CODE HERE
import sys

def solve():
    # Read the initial string S and target string T from standard input
    S_str = sys.stdin.readline().strip()
    T_str = sys.stdin.readline().strip()
    
    # Convert strings to lists of characters for easier modification.
    # Python strings are immutable, lists are mutable.
    S = list(S_str) 
    T = list(T_str) 
    N = len(S) # Store the length of the strings (they have equal length)

    # Handle the edge case where S is already equal to T.
    # In this case, 0 operations are needed and the array X is empty.
    if S == T:
        print(0) # Output 0 steps
        return # Exit the function

    # Create a mutable copy of S to represent the current state of the string.
    current_S = list(S) 
    
    # Initialize an empty list X to store the sequence of generated strings.
    X = [] 

    # Loop continues as long as the current string `current_S` is not equal to the target string `T`.
    # List comparison `current_S != T` compares elements pairwise efficiently.
    while current_S != T: 
        
        # Find all indices where the characters in `current_S` and `T` differ.
        # These are the candidate positions for the next change operation.
        diff_indices = []
        for k in range(N):
            if current_S[k] != T[k]:
                diff_indices.append(k)
        
        # Determine the best index `k` to change based on the greedy strategy
        # required to produce the lexicographically smallest sequence X.
        best_k = -1 # Initialize best_k to an invalid index

        # Strategy Rule 1 (derived as Case 3 in thought process):
        # Check if there exists any differing index `k` such that the target character T[k] 
        # is lexicographically smaller than the current character current_S[k].
        min_k_less = -1 # Track the minimum index k satisfying T[k] < current_S[k]
        for k in diff_indices:
            if T[k] < current_S[k]:
                # If this is the first such index found, or if k is smaller than the current minimum found so far
                if min_k_less == -1 or k < min_k_less:
                    min_k_less = k
        
        if min_k_less != -1:
            # If Rule 1 applies (at least one such k found), choose the smallest such index `k`.
            # This choice ensures the resulting string is lexicographically as small as possible among options that decrease the character value at the earliest possible position.
            best_k = min_k_less
        else:
            # Strategy Rule 2 (derived as Case 4 in thought process):
            # If Rule 1 does not apply, it means for all differing indices `k`, T[k] > current_S[k].
            # In this scenario, to minimize the resulting string lexicographically,
            # we must choose the largest index `k` among the differing positions.
            max_k_greater = -1 # Track maximum index k among differing positions
            # Iterate through all differing indices to find the maximum one.
            for k in diff_indices:
                 # Check if current k is larger than previously found max index.
                 # The check `max_k_greater == -1` handles the initial state when no max has been found yet.
                 if max_k_greater == -1 or k > max_k_greater:
                     max_k_greater = k
            # Choose this largest index.
            best_k = max_k_greater

        # Apply the change to `current_S` at the determined best index `best_k`.
        # The character at this position is updated to match the target string T.
        current_S[best_k] = T[best_k]
        
        # Convert the modified list `current_S` back to a string and append it to the result list X.
        X.append("".join(current_S)) 

    # After the loop terminates (meaning current_S == T), print the results.
    # First line: Print the total number of steps `M`, which is the number of elements in X.
    print(len(X))
    # Subsequent lines: Print each string generated and stored in X, one per line.
    for s_step in X:
        print(s_step)

# Execute the solve function when the script is run.
solve()
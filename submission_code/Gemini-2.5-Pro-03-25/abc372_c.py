# YOUR CODE HERE
import sys

def solve():
    # Read N (length of string) and Q (number of queries) from the first line of input
    N, Q = map(int, sys.stdin.readline().split())
    
    # Read the initial string S and convert it to a list of characters.
    # Using a list allows for efficient modification of characters at specific indices.
    S_list = list(sys.stdin.readline().strip())

    # Calculate the initial count of "ABC" substrings in S.
    # Iterate through all possible starting positions for a substring of length 3.
    count = 0
    for i in range(N - 2):
        # Check if the substring starting at index i is equal to "ABC"
        if S_list[i] == 'A' and S_list[i+1] == 'B' and S_list[i+2] == 'C':
            count += 1

    # Define a helper function to check if the substring "ABC" starts at a given index `idx`.
    # This function uses the `S_list` from the outer scope, reflecting the current state of the string.
    def check(idx):
        # Check boundary conditions: The indices idx, idx+1, and idx+2 must all be valid indices
        # within the string bounds [0, N-1]. This implies idx must be non-negative and idx+2 must be less than N.
        if 0 <= idx and idx + 2 < N:
            # Check if the characters at these three consecutive positions form the string "ABC".
            return S_list[idx] == 'A' and S_list[idx+1] == 'B' and S_list[idx+2] == 'C'
        # If the indices are out of bounds for a length-3 substring, it cannot be "ABC".
        return False

    # Process each of the Q queries sequentially.
    for _ in range(Q):
        # Read the query parameters: position X_i (1-based) and the new character C_i.
        line = sys.stdin.readline().split()
        X = int(line[0])  # 1-based index from input
        C = line[1]       # The new character to place at position X
        
        # Convert the 1-based index X from the input to a 0-based index k for list access.
        k = X - 1 

        # Optimization: If the character at the target position k is already the new character C,
        # the string does not actually change. In this case, the count of "ABC" remains the same.
        # Print the current count and proceed to the next query without further computation.
        if S_list[k] == C:
            print(count)
            continue

        # Identify the potential starting indices of "ABC" substrings that could be affected by changing the character at index k.
        # An "ABC" substring starting at index `idx` involves characters at indices `idx`, `idx+1`, `idx+2`.
        # Changing the character at index `k` can potentially affect this substring if `k` is one of these three indices.
        # This means the starting index `idx` of an affected "ABC" could be `k`, `k-1`, or `k-2`.
        # These are the only starting positions we need to re-evaluate.
        affected_start_indices = [k - 2, k - 1, k]
        
        # Check the state BEFORE the character change:
        # Iterate through the potentially affected starting indices.
        # For each index `idx`, if `check(idx)` returns True BEFORE the change, it means an "ABC" instance starting at `idx` exists.
        # This instance involves the character at `k` and will potentially be destroyed by the modification.
        # Decrement the count for each such instance found.
        for idx in affected_start_indices:
            if check(idx):
                 count -= 1 
        
        # Perform the actual character update in the list representation of the string.
        # Replace the character at index k with the new character C.
        S_list[k] = C
        
        # Check the state AFTER the character change:
        # Iterate through the potentially affected starting indices again.
        # For each index `idx`, if `check(idx)` returns True AFTER the change, it means an "ABC" instance starting at `idx` now exists.
        # This could be a newly formed instance or an instance that was destroyed and then reformed by the change.
        # Increment the count for each such instance found.
        for idx in affected_start_indices:
             if check(idx):
                 count += 1 
        
        # After adjusting the count based on destroyed and created/reformed "ABC" instances,
        # print the final count for the current query.
        print(count)

# Execute the main function `solve()` to run the solution.
solve()
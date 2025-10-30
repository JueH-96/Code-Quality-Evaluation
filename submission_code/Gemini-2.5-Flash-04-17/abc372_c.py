# YOUR CODE HERE
import sys

# Use fast I/O
input = sys.stdin.readline

def solve():
    # Read N and Q
    N, Q = map(int, input().split())
    # Read the initial string S and convert it to a list of characters
    S = list(input().strip()) 
    
    # Calculate the initial count of "ABC" substrings
    total_count = 0
    # Iterate through all possible start indices k for a length-3 substring
    # These indices range from 0 up to N-3
    for k in range(N - 2):
        # Check if the substring starting at k is "ABC"
        if S[k] == 'A' and S[k+1] == 'B' and S[k+2] == 'C':
            total_count += 1
    
    # Process each query
    for _ in range(Q):
        # Read query: index X (1-based) and character C
        X, C = input().split()
        X = int(X)
        i = X - 1 # Convert the 1-based index X to a 0-based index i for Python list
        
        old_char = S[i] # Store the original character at index i
        new_char = C    # The new character to replace it with
        
        # If the character doesn't change, the count of "ABC" remains the same.
        # We can optimize by skipping the rest of the logic for this query.
        if old_char == new_char:
            print(total_count)
            continue
            
        # Calculate the change in "ABC" count (delta) resulting from this query
        delta = 0
        
        # A change at index i can only affect "ABC" substrings that contain index i.
        # An "ABC" substring starts at index j (0 <= j <= N-3) and covers indices j, j+1, j+2.
        # This substring contains index i if j <= i <= j+2, which means i-2 <= j <= i.
        # So, the potential start indices j that are affected are i-2, i-1, and i.
        # We only need to consider these potential start indices j that are also valid
        # start indices within the string bounds, i.e., 0 <= j <= N-3.
        # The relevant range of valid start indices j is thus [max(0, i-2), min(N-3, i)].
        
        start_j = max(0, i - 2)
        end_j = min(N - 3, i) # This is the inclusive end index for j
        
        # Check potentially affected "ABC"s before the change S[i] -> new_char
        # Iterate through the relevant valid start indices j from start_j up to end_j (inclusive).
        # The range function goes up to (stop - 1), so we use end_j + 1 as the stop value.
        for j in range(start_j, end_j + 1):
            
            # The indices within the substring S[j:j+3] are k0=j, k1=j+1, k2=j+2.
            k0, k1, k2 = j, j+1, j+2
            
            # Get the characters at k0, k1, k2 *before* the change.
            # If the index k is the changed index i, use old_char; otherwise, use the current S[k].
            char_k0_before = old_char if k0 == i else S[k0]
            char_k1_before = old_char if k1 == i else S[k1]
            char_k2_before = old_char if k2 == i else S[k2]
            
            # If the substring was "ABC" before the change, decrement the delta count.
            if char_k0_before == 'A' and char_k1_before == 'B' and char_k2_before == 'C':
                delta -= 1

        # Apply the character change to the string list S.
        S[i] = new_char
        
        # Check potentially affected "ABC"s after the change
        # Iterate through the same range of relevant valid start indices j.
        for j in range(start_j, end_j + 1):
             
            # After change, the characters are simply S[j], S[j+1], and S[j+2].
            # S[i] has already been updated to new_char.
            
            # If the substring S[j:j+3] is now "ABC" after the change, increment the delta count.
            if S[j] == 'A' and S[j+1] == 'B' and S[j+2] == 'C':
                delta += 1

        # Update the total count by the calculated delta.
        total_count += delta
        
        # Print the updated total count for the current query.
        print(total_count)

# Execute the solve function to run the program
solve()
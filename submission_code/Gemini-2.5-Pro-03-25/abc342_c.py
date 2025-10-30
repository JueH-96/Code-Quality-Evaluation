import sys

# Function to solve the problem
def solve():
    # Read N, the length of the string
    N = int(sys.stdin.readline())
    # Read the string S
    S = sys.stdin.readline().strip()
    # Read Q, the number of operations
    Q = int(sys.stdin.readline())

    # Initialize the mapping. `map_to[i]` will store the character that
    # the i-th letter of the alphabet (0='a', 1='b', ...) eventually maps to
    # after all operations processed so far.
    # We use a list of characters for this mapping. The index corresponds to the
    # ASCII value offset from 'a'. For example, index 0 corresponds to 'a', 1 to 'b', etc.
    # Initially, each character maps to itself.
    map_to = list("abcdefghijklmnopqrstuvwxyz") 

    # Process each of the Q operations
    for _ in range(Q):
        # Read the characters c_i and d_i for the current operation.
        # The operation is: replace all occurrences of character c with character d.
        c, d = sys.stdin.readline().split()
        
        # We implement this operation by updating our character mapping `map_to`.
        # The logic is: any original character `orig_char` that currently maps to `c` 
        # (i.e., `map_to[ord(orig_char) - ord('a')] == c`) should now map to `d`.
        
        # We iterate through all 26 possible original characters ('a' through 'z').
        # The index `i` corresponds to the original character `chr(ord('a') + i)`.
        for i in range(26):
            # Check if the character that `chr(ord('a') + i)` currently maps to is `c`.
            if map_to[i] == c:
                # If it maps to `c`, we update its mapping to `d`.
                # This means that after this operation, the original character `chr(ord('a') + i)`
                # will eventually resolve to `d` (potentially via intermediate steps involving `c`).
                map_to[i] = d

    # After processing all Q operations, the `map_to` list contains the final mapping
    # from each original character ('a' through 'z') to its final form.

    # Construct the final string based on the original string S and the final mapping.
    # We build the result string character by character by looking up the final mapped value
    # for each character in the original string S.
    # Using a list to collect characters and then joining is efficient in Python.
    final_S_list = []
    for char_in_S in S:
        # Calculate the index (0-25) corresponding to the character `char_in_S`.
        # This index represents the original character in our `map_to` list.
        original_char_idx = ord(char_in_S) - ord('a')
        
        # Look up the final character this original character maps to using the `map_to` list.
        final_char = map_to[original_char_idx]
        
        # Append the final character to our result list.
        final_S_list.append(final_char)
    
    # Join the characters in the list to form the final result string.
    # Print the final string to standard output.
    print("".join(final_S_list))

# Execute the solve function to run the program logic
solve()
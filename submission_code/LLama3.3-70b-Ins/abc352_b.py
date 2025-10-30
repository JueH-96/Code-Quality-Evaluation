def find_correctly_typed_characters(S, T):
    """
    This function determines the positions in T of the correctly typed characters.
    
    Parameters:
    S (str): The string that Takahashi tried to type.
    T (str): The actual string typed by Takahashi.
    
    Returns:
    list: A list of positions of correctly typed characters in T.
    """
    
    # Initialize an empty list to store the positions of correctly typed characters
    correctly_typed_positions = []
    
    # Initialize two pointers, one for S and one for T
    s_pointer = 0
    t_pointer = 0
    
    # Loop through T
    while t_pointer < len(T):
        # If the current characters in S and T are the same
        if s_pointer < len(S) and S[s_pointer] == T[t_pointer]:
            # Add the current position in T to the list of correctly typed positions
            correctly_typed_positions.append(t_pointer + 1)
            # Move the pointer for S forward
            s_pointer += 1
        # Move the pointer for T forward
        t_pointer += 1
    
    # Return the list of correctly typed positions
    return correctly_typed_positions


# Read the inputs from stdin
S = input()
T = input()

# Find the positions of correctly typed characters
correctly_typed_positions = find_correctly_typed_characters(S, T)

# Print the positions of correctly typed characters
print(' '.join(map(str, correctly_typed_positions)))
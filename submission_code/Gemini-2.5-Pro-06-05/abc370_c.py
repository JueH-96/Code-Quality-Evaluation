import sys

def solve():
    """
    Reads two strings S and T and finds the lexicographically smallest sequence
    of transformations from S to T, one character change at a time.
    """
    # Read the input strings from standard input.
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()

    # If S is already equal to T, no operations are needed.
    if s == t:
        print(0)
        return

    # Get the length of the strings.
    n = len(s)
    
    # Use lists of characters for efficient modification.
    current_s_list = list(s)
    t_list = list(t)
    
    # This will store the resulting array of strings X.
    result_x = []

    # Loop until the current string equals the target string.
    while current_s_list != t_list:
        # Find all indices where the current string and target string differ.
        diff_indices = [i for i in range(n) if current_s_list[i] != t_list[i]]

        # Generate candidate lists of characters for the next step.
        # Each candidate is formed by fixing one of the differing characters.
        candidate_lists = []
        for idx in diff_indices:
            # Create a copy of the current list of characters.
            temp_s_list = current_s_list[:]
            # Fix the character at the chosen index to match the target.
            temp_s_list[idx] = t_list[idx]
            candidate_lists.append(temp_s_list)

        # To find the lexicographically smallest path, we make a greedy choice.
        # We select the candidate that is lexicographically smallest. Python's min()
        # on a list of lists performs this lexicographical comparison correctly.
        best_next_s_list = min(candidate_lists)

        # Append the chosen string to our result array X.
        result_x.append("".join(best_next_s_list))
        
        # Update the current state for the next iteration.
        current_s_list = best_next_s_list

    # Print the total number of operations (the size of X).
    print(len(result_x))
    
    # Print each string in the generated array X.
    for item in result_x:
        print(item)

# Run the solution.
solve()
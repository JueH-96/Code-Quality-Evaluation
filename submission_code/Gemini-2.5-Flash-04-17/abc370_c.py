import sys

def solve():
    # Read input strings S and T
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    N = len(S)
    
    # Initialize current string state as a list for mutability
    current_S_list = list(S)
    # Initialize the list to store intermediate strings
    X = []
    
    # Loop until current string equals target string
    while "".join(current_S_list) != T:
        best_next_S_str = None
        
        # Iterate through all possible single-character changes
        # Find the lexicographically smallest string reachable in one step
        for i in range(N):
            original_char = current_S_list[i]
            
            for char_code in range(ord('a'), ord('z') + 1):
                c = chr(char_code)
                
                # Apply the potential change
                current_S_list[i] = c
                next_S_str = "".join(current_S_list)
                
                # Update best_next_S_str if this new string is smaller
                if best_next_S_str is None or next_S_str < best_next_S_str:
                    best_next_S_str = next_S_str
                
                # Restore the original character to explore other changes from current_S
                current_S_list[i] = original_char

        # Now that the lexicographically smallest next string (best_next_S_str) is found
        # Apply this best change to the current_S_list permanently
        # We need to find which single character change (i, c) results in best_next_S_str
        # Since we iterated through i then c when finding best_next_S_str,
        # the first (i, c) pair that produces it is the one we want (for tie-breaking if multiple changes result in the same smallest string).
        applied_change = False
        for i in range(N):
            original_char = current_S_list[i]
            for char_code in range(ord('a'), ord('z') + 1):
                c = chr(char_code)
                current_S_list[i] = c
                
                # Check if this change yields the best string found
                if "".join(current_S_list) == best_next_S_str:
                    # This is the change we apply for this step
                    applied_change = True
                    break # Found the correct character c for this index i
                    
                # If this char didn't result in the best string, restore and try next char
                current_S_list[i] = original_char 
                
            if applied_change:
                break # Found the correct index i

        # The current_S_list has now been updated to best_next_S_str
        # Append the resulting string to the list X
        X.append("".join(current_S_list))

    # Output the number of elements in X
    print(len(X))
    # Output each string in X on a new line
    for s in X:
        print(s)

solve()
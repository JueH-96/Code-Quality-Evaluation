import sys

def solve():
    # Read S and T from standard input.
    # .strip() removes leading/trailing whitespace, including the newline character.
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    s_len = len(S)
    # t_len = len(T) # Not strictly needed for the loop logic due to problem guarantee

    s_ptr = 0  # Pointer for S, indicating the character we are currently trying to type.
    t_ptr = 0  # Pointer for T, indicating the character currently being processed in T.

    result_indices = [] # List to store the 1-based indices of correctly typed characters.

    # Loop until all characters of S have been successfully typed.
    # s_ptr will range from 0 to s_len-1. When s_ptr becomes s_len, all characters are found.
    while s_ptr < s_len:
        # The problem guarantees that T is formed according to the rules.
        # This implies that for the current S[s_ptr], there will be a matching T[t_ptr]
        # eventually (possibly after some mismatches).
        # Thus, t_ptr will not go out of bounds of T (i.e., t_ptr < len(T))
        # before S[s_ptr] is found and matched. So, T[t_ptr] access is safe.
        
        if S[s_ptr] == T[t_ptr]:
            # Correctly typed character found.
            # Add its 1-based position in T to our results.
            result_indices.append(t_ptr + 1)
            
            # Move to the next character in S that Takahashi intends to type.
            s_ptr += 1
            
            # Move to the next character in the actual typed string T.
            t_ptr += 1
        else:
            # Mistakenly typed character. T[t_ptr] is the mistake.
            # Takahashi is still trying to type S[s_ptr]. So, s_ptr does not change.
            
            # Move to the next character in the actual typed string T.
            t_ptr += 1
            
    # Print the collected indices.
    # The `*` operator unpacks the list `result_indices` into individual arguments
    # for the print function, which then prints them separated by spaces by default.
    print(*(result_indices))

# Standard boilerplate to call the main solve function.
if __name__ == '__main__':
    solve()
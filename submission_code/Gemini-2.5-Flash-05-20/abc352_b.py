# YOUR CODE HERE
import sys

def solve():
    # Read string S from standard input
    S = sys.stdin.readline().strip()
    # Read string T from standard input
    T = sys.stdin.readline().strip()

    # Initialize pointers for string S and string T
    s_ptr = 0  # Pointer for current character in S that we are looking for
    t_ptr = 0  # Pointer for current character in T being examined

    # List to store the 1-indexed positions of correctly typed characters
    correct_positions = []

    # Iterate while there are characters left in S to find and characters left in T to examine
    # The loop continues as long as we haven't found all characters of S
    # and we still have characters in T to check.
    while s_ptr < len(S) and t_ptr < len(T):
        # If the character in T matches the character we are currently looking for in S
        if S[s_ptr] == T[t_ptr]:
            # This character in T is a correctly typed character
            correct_positions.append(t_ptr + 1) # Store its 1-indexed position
            s_ptr += 1  # Move to the next character in S, as we found the current one
        
        # In either case (match or mismatch), we advance the pointer in T.
        # If it was a match, we just processed T[t_ptr] and move to the next char in T.
        # If it was a mismatch, T[t_ptr] was a mistakenly typed character,
        # which means we need to skip it and continue searching for S[s_ptr]
        # in the subsequent characters of T.
        t_ptr += 1
    
    # After the loop, correct_positions will contain all the 1-indexed positions
    # of the correctly typed characters from T that form S.
    # Print the positions, space-separated.
    # The '*' operator unpacks the list into separate arguments for print,
    # which by default are printed space-separated.
    print(*correct_positions)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()
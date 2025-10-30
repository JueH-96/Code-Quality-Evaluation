# YOUR CODE HERE
import sys

def solve():
    # Read the target string S from standard input
    s = sys.stdin.readline().strip()
    # Read the actually typed string T from standard input
    t = sys.stdin.readline().strip()
    
    # Get the lengths of the strings S and T
    s_len = len(s)
    t_len = len(t)
    
    # Initialize an empty list to store the 1-based indices 
    # of the correctly typed characters in T
    result_indices = []
    
    # Initialize pointers:
    # s_idx tracks the current index in S, representing the character we are looking for.
    # t_idx tracks the current index in T, representing the character we are examining.
    s_idx = 0  
    t_idx = 0  
    
    # Iterate through the typed string T using the t_idx pointer.
    # The loop continues as long as we haven't found all characters of S (s_idx < s_len)
    # and we haven't exhausted the typed string T (t_idx < t_len).
    while s_idx < s_len and t_idx < t_len:
        # Check if the character at the current position in T (T[t_idx])
        # matches the target character from S (S[s_idx]).
        if t[t_idx] == s[s_idx]:
            # If they match, this character T[t_idx] must be the correctly typed instance
            # of the character S[s_idx]. According to the problem description, mistakes
            # are always different characters. Once the correct character is typed,
            # the user moves on to the next character in S.
            # Therefore, we record the 1-based position (t_idx + 1) of this character.
            result_indices.append(t_idx + 1)
            # Since we found the character S[s_idx], we advance the s_idx pointer
            # to look for the next character in S.
            s_idx += 1
        
        # Regardless of whether a match was found or not, we always advance the t_idx pointer
        # to examine the next character in T.
        # If T[t_idx] matched S[s_idx], we move past it to start searching for S[s_idx+1].
        # If T[t_idx] did not match S[s_idx], it must be a mistake made while trying to type S[s_idx].
        # We skip this mistaken character and continue searching for S[s_idx] at the next position in T.
        t_idx += 1
        
    # After the loop finishes, the result_indices list contains the 1-based indices 
    # of all correctly typed characters from S within T, in ascending order.
    # Print these indices separated by spaces. The * operator unpacks the list 
    # elements as separate arguments to the print function.
    print(*(result_indices))

# Execute the solve function which encapsulates the main logic.
solve()
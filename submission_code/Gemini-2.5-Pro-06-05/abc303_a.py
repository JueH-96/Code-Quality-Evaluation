# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the similar strings problem, and prints the output.
    """
    try:
        # Read the integer N from the first line of input.
        N = int(sys.stdin.readline())
        # Read the string S from the second line of input.
        S = sys.stdin.readline().strip()
        # Read the string T from the third line of input.
        T = sys.stdin.readline().strip()
    except (ValueError, IndexError):
        # Handle potential malformed input, though not expected by problem constraints.
        return

    # Iterate through the strings from the first character to the last.
    for i in range(N):
        char_s = S[i]
        char_t = T[i]

        # Check for similarity conditions.
        # Condition 1: The characters are identical.
        if char_s == char_t:
            continue
        
        # Condition 2: One is '1' and the other is 'l'.
        # Using a set comparison is a concise way to check this symmetric relationship.
        if {char_s, char_t} == {'1', 'l'}:
            continue
            
        # Condition 3: One is '0' and the other is 'o'.
        if {char_s, char_t} == {'0', 'o'}:
            continue
        
        # If none of the similarity conditions are met for the current pair of characters,
        # the strings are not similar. Print "No" and terminate the program.
        print("No")
        return

    # If the loop completes, it means all character pairs were similar.
    # Therefore, the strings are similar.
    print("Yes")

# Run the solution
solve()
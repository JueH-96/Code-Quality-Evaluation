# YOUR CODE HERE
import sys
# Import permutations function from itertools module
from itertools import permutations

# Function to check if a string `p_str` of length `N` contains 
# any palindrome substring of length `K`.
def check(p_str, N, K):
    """
    Checks if string p_str of length N contains a palindrome substring of length K.
    Returns True if such a substring exists, False otherwise.
    """
    # Iterate through all possible starting indices `i` for a substring of length K.
    # The indices range from 0 to N-K (inclusive).
    for i in range(N - K + 1):
        # Extract the substring of length K starting at index i.
        substring = p_str[i : i+K]
        
        # Check if the extracted substring is a palindrome.
        # A string is a palindrome if it is equal to its reverse.
        # Python slice `[::-1]` efficiently creates a reversed copy of the string.
        if substring == substring[::-1]: 
            # If a palindrome substring of length K is found, we can immediately conclude
            # that the string `p_str` contains such a palindrome.
            # Return True to indicate a palindrome was found.
            return True 
            
    # If the loop finishes without finding any palindrome substring of length K,
    # it means the string `p_str` does not contain any such substring.
    # Return False to indicate no palindrome of length K was found.
    return False 

# Main function to solve the problem
def solve():
    # Read N and K from the first line of standard input.
    # N is the length of the string, K is the length of the palindrome to check for.
    # `map(int, ...)` converts the split strings into integers.
    N, K = map(int, sys.stdin.readline().split())
    
    # Read the string S from the second line of standard input.
    # `.strip()` removes leading/trailing whitespace, including the potential newline character.
    S = sys.stdin.readline().strip()

    # Initialize a counter for the number of valid permutations found.
    # A permutation is valid if it does not contain a palindrome substring of length K.
    valid_permutation_count = 0
    
    # Use a set data structure to store the distinct permutations of S.
    # `itertools.permutations` generates all N! permutations. If S has repeated characters,
    # some permutations will be identical. Using a set automatically filters out duplicates.
    distinct_permutations = set()
    
    # Generate all possible permutations of the characters in string S.
    # `permutations(S)` returns an iterator yielding tuples of characters.
    # For example, if S = 'aab', it yields ('a', 'a', 'b'), ('a', 'b', 'a'), ('a', 'a', 'b'), ('a', 'b', 'a'), ('b', 'a', 'a'), ('b', 'a', 'a').
    for p_tuple in permutations(S):
        # Join the characters in the tuple `p_tuple` to form a string permutation.
        p_str = "".join(p_tuple)
        # Add the generated string permutation to the set. If it's already present, the set structure handles it efficiently.
        distinct_permutations.add(p_str)

    # Iterate through each unique string permutation stored in the set `distinct_permutations`.
    for p_str in distinct_permutations:
        # Check if the current permutation string `p_str` contains a palindrome substring of length K
        # using the helper function `check`.
        # `check(p_str, N, K)` returns True if it finds such a palindrome, False otherwise.
        # We are interested in counting permutations that do *not* contain such a palindrome.
        if not check(p_str, N, K):
             # If `check` returns False, it means no palindrome substring of length K was found in `p_str`.
             # This permutation satisfies the condition specified in the problem.
             # Increment the counter for valid permutations.
             valid_permutation_count += 1

    # After iterating through all distinct permutations and checking each one,
    # print the final count of valid permutations to standard output.
    print(valid_permutation_count)

# Call the `solve` function to execute the program logic when the script is run.
solve()
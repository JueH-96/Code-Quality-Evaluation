# YOUR CODE HERE
import sys

# Function to check if a string 'sub' is a subsequence of 'main'
# This function iterates through 'main' at most once, making it efficient (O(len(main))).
def is_subsequence(sub, main):
    """
    Checks if 'sub' is a subsequence of 'main'.
    A subsequence is formed by deleting zero or more characters from 'main' 
    without changing the order of the remaining characters.

    Args:
        sub (str): The potential subsequence string.
        main (str): The main string to search within.

    Returns:
        bool: True if 'sub' is a subsequence of 'main', False otherwise.
    
    Example: is_subsequence("ace", "abcde") -> True
    Example: is_subsequence("aec", "abcde") -> False
    """
    # Create an iterator for the main string. This allows consuming characters one by one.
    it = iter(main)
    
    # Use the 'all()' function with a generator expression.
    # For each character 'c' in the subsequence 'sub':
    # Check if 'c' can be found in the *remaining* part of the iterator 'it'.
    # The 'in' operator used with an iterator (`c in it`) searches for 'c' starting 
    # from the current position of the iterator. If found, it consumes characters 
    # up to and including 'c'. If not found before the iterator is exhausted, it returns False.
    # 'all()' returns True only if 'c in it' is True for *every* character 'c' in 'sub'.
    # If 'sub' is empty, all([]) returns True, which is correct.
    return all(c in it for c in sub)


def solve():
    # Read the input string S (lowercase) from standard input
    s = sys.stdin.readline().strip()
    # Read the input airport code T (uppercase, length 3) from standard input
    t_upper = sys.stdin.readline().strip()
    
    # Convert the target airport code T to lowercase. 
    # This allows direct comparison with characters in S.
    t_lower = t_upper.lower()

    # --- Check Condition 1 ---
    # T is an airport code if it can be derived from a subsequence of length 3 from S,
    # converted to uppercase.
    # This is equivalent to checking if the lowercase version of T (t_lower) 
    # is a subsequence of S.
    found1 = is_subsequence(t_lower, s)

    # --- Check Condition 2 ---
    # T is an airport code if it can be derived from a subsequence of length 2 from S,
    # converted to uppercase, and appended with 'X'.
    found2 = False
    # This condition is only possible if the third character of T is 'X'.
    if t_upper[2] == 'X':
        # If T ends with 'X', we need to check if the first two characters of T (in lowercase)
        # form a subsequence of S.
        t_prefix = t_lower[:2] # Get the first two characters, e.g., "la" from "lax"
        found2 = is_subsequence(t_prefix, s)

    # --- Determine the final answer ---
    # T is a valid airport code for S if either Condition 1 or Condition 2 is met.
    if found1 or found2:
        print("Yes")
    else:
        # If neither condition is met, T is not a valid airport code for S.
        print("No")

# Execute the main logic of the program
solve()
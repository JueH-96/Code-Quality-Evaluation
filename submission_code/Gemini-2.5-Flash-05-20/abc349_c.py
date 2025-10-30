import sys

# Function to check if target_str is a subsequence of source_str (case-insensitive for source_str)
def check_subsequence(source_str, target_str):
    """
    Determines if target_str is a subsequence of source_str.
    Characters in source_str are treated as uppercase for comparison.
    
    Args:
        source_str (str): The string to search within.
        target_str (str): The subsequence to find.
        
    Returns:
        bool: True if target_str is a subsequence of source_str, False otherwise.
    """
    target_idx = 0
    source_idx = 0
    
    # Iterate through the source string
    while source_idx < len(source_str) and target_idx < len(target_str):
        # If the current character in source_str (converted to uppercase) matches
        # the current character in target_str, move to the next character in target_str.
        if source_str[source_idx].upper() == target_str[target_idx]:
            target_idx += 1
        # Always move to the next character in source_str
        source_idx += 1
            
    # If target_idx equals the length of target_str, it means all characters
    # of target_str were found in order within source_str.
    return target_idx == len(target_str)

def solve():
    # Read input strings S and T from standard input
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # Rule 1 Check:
    # T is an airport code if it's formed by a subsequence of length 3 from S,
    # converted to uppercase. This is directly checking if T is a subsequence of S.
    rule1_applies = check_subsequence(S, T)

    # Rule 2 Check:
    # T is an airport code if it's formed by a subsequence of length 2 from S,
    # converted to uppercase, and then 'X' is appended.
    rule2_applies = False
    # This rule only applies if the third character of T is 'X'.
    if T[2] == 'X':
        # We need to check if the first two characters of T (T[0]T[1])
        # form a subsequence of S.
        rule2_applies = check_subsequence(S, T[0:2])

    # Print "Yes" if either Rule 1 or Rule 2 applies, otherwise print "No".
    if rule1_applies or rule2_applies:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the program
solve()
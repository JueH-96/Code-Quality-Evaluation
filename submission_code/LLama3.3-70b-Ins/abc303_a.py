def are_similar_strings(S, T):
    """
    Checks if two strings S and T are similar strings.
    
    Args:
        S (str): The first string.
        T (str): The second string.
    
    Returns:
        bool: True if S and T are similar strings, False otherwise.
    """
    # Define similar characters
    similar_chars = {
        '1': ['1', 'l'],
        'l': ['1', 'l'],
        '0': ['0', 'o'],
        'o': ['0', 'o']
    }
    
    # Iterate over characters in S and T
    for s, t in zip(S, T):
        # If characters are the same, continue
        if s == t:
            continue
        # If characters are not similar, return False
        if s not in similar_chars or t not in similar_chars.get(s, []):
            return False
    
    # If all characters are similar, return True
    return True

# Read input from stdin
N = int(input())
S = input()
T = input()

# Check if S and T are similar strings
if are_similar_strings(S, T):
    print("Yes")
else:
    print("No")
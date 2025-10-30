def solve(W, B):
    """
    Checks if there is a substring of S that consists of W occurrences of w and B occurrences of b.
    
    Args:
        W (int): The number of white keys.
        B (int): The number of black keys.
    
    Returns:
        str: "Yes" if a substring exists, "No" otherwise.
    """
    # Generate the string S by repeating the pattern "wbwbwwbwbwbw"
    S = "wbwbwwbwbwbw" * 100
    
    # Iterate over all possible substrings of S
    for i in range(len(S)):
        for j in range(i + 1, len(S) + 1):
            substring = S[i:j]
            
            # Count the occurrences of 'w' and 'b' in the current substring
            w_count = substring.count('w')
            b_count = substring.count('b')
            
            # Check if the current substring meets the conditions
            if w_count == W and b_count == B:
                return "Yes"
    
    # If no substring meets the conditions, return "No"
    return "No"

# Read the inputs from stdin
W, B = map(int, input().split())

# Solve the problem and write the answer to stdout
print(solve(W, B))
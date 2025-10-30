import sys

# Function to calculate length of longest common prefix
def get_prefix_len(S, T_prime):
    """Calculates the length of the longest common prefix of S and T_prime."""
    LS = len(S)
    LT = len(T_prime)
    k = 0
    # Iterate while indices are within bounds for both strings and characters match
    while k < LS and k < LT and S[k] == T_prime[k]:
        k += 1
    return k

# Function to calculate length of longest common suffix
def get_suffix_len(S, T_prime):
    """Calculates the length of the longest common suffix of S and T_prime."""
    LS = len(S)
    LT = len(T_prime)
    k = 0
    # Iterate while indices are valid (non-negative) and characters match
    # Check k < LS and k < LT to ensure indices S[LS-1-k] and T_prime[LT-1-k] are valid for comparison
    while k < LS and k < LT and S[LS - 1 - k] == T_prime[LT - 1 - k]:
        k += 1
    return k

# Function to check if S could be the original string T based on T_prime
def check_candidate(S, T_prime):
    """Checks if S could be the original string T, given the received string T_prime.
    
    This function evaluates the four possible conditions:
    1. T' = T (no change)
    2. T' obtained by inserting one character into T
    3. T' obtained by deleting one character from T
    4. T' obtained by changing one character in T
    
    It returns True if S could be T under any of these conditions, False otherwise.
    """
    LS = len(S)
    LT = len(T_prime)

    # If lengths differ by more than 1, S cannot be T according to the problem rules.
    # The allowed operations change length by at most 1.
    if abs(LS - LT) > 1:
        return False

    # Case 1: Strings have the same length.
    # Possibilities: T' = T (no change) or T' differs from T by one character change.
    if LS == LT:
        # Check condition 1: T' = T (no change)
        if S == T_prime: 
             return True
        
        # If not identical, check condition 4 (one character changed)
        # This requires the strings to differ at exactly one position.
        # We can efficiently check this using prefix and suffix lengths.
        prefix_len = get_prefix_len(S, T_prime)
        suffix_len = get_suffix_len(S, T_prime)
        # A single character change means the non-matching part has length 1.
        # This occurs if the combined length of the matching prefix and suffix is exactly LS - 1.
        # For example, if S = "apple", T' = "apply", prefix="appl"(len 4), suffix=""(len 0). LS=5. 4+0 = 5-1. Correct.
        # If S = "abcde", T' = "abXde", prefix="ab"(len 2), suffix="de"(len 2). LS=5. 2+2 = 5-1. Correct.
        if prefix_len + suffix_len == LS - 1:
            return True
            
    # Case 2: T' is longer than S by 1 character.
    # Possibility: T' was obtained by inserting one character into T=S.
    # This means S must be a subsequence of T' obtainable by deleting one character.
    elif LS == LT - 1: 
        # Calculate prefix and suffix lengths.
        prefix_len = get_prefix_len(S, T_prime)
        suffix_len = get_suffix_len(S, T_prime)
        # S is obtainable by deleting one character from T' if the matching prefix and suffix cover all of S.
        # For example, S = "apple", T' = "dapple". prefix=""(len 0), suffix="apple"(len 5). LS=5. 0+5 >= 5. Correct.
        # S = "apple", T' = "appdle". prefix="app"(len 3), suffix="le"(len 2). LS=5. 3+2 >= 5. Correct.
        if prefix_len + suffix_len >= LS:
            return True

    # Case 3: T' is shorter than S by 1 character.
    # Possibility: T' was obtained by deleting one character from T=S.
    # This means T' must be a subsequence of S obtainable by deleting one character.
    elif LS == LT + 1: 
        # Calculate prefix and suffix lengths.
        prefix_len = get_prefix_len(S, T_prime)
        suffix_len = get_suffix_len(S, T_prime)
        # T' is obtainable by deleting one character from S if the matching prefix and suffix cover all of T'.
        # For example, S = "apple", T' = "aple". prefix="ap"(len 2), suffix="le"(len 2). LT=4. 2+2 >= 4. Correct.
        # S = "dapple", T' = "apple". prefix=""(len 0), suffix="apple"(len 5). LT=5. 0+5 >= 5. Correct.
        if prefix_len + suffix_len >= LT:
            return True

    # If none of the specific conditions derived from the problem statement match.
    return False

# Main solver function
def solve():
    """Reads input, checks each string S_i against T', and prints the result."""
    # Read N and T' from the first line of input using sys.stdin for efficiency
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    T_prime = line1[1]
    
    # List to store 1-based indices of candidate strings
    ans_indices = []

    # Iterate through N strings S_i
    for i in range(1, N + 1):
        # Read the i-th string S_i using sys.stdin and strip potential whitespace
        S = sys.stdin.readline().strip()
        # Check if S_i could potentially be the original string T
        if check_candidate(S, T_prime):
            # If it's a candidate, add its 1-based index to the list
            ans_indices.append(i)

    # Print the total count of candidate strings found
    print(len(ans_indices))
    # Print the space-separated list of indices if any exist. 
    # Convert indices to strings for printing.
    if ans_indices: 
        print(*(map(str, ans_indices)))

# Execute the solver function when the script runs
solve()
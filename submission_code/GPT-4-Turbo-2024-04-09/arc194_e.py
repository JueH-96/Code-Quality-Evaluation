def can_transform(N, X, Y, S, T):
    # We will use a greedy approach to check if we can transform S into T
    # by looking for patterns from left to right and applying operations when needed.
    
    # We need to check if we can transform S into T by swapping blocks of X zeros followed by Y ones
    # and vice versa.
    
    # We will iterate over the strings and whenever we find a mismatch, we will try to resolve it
    # using the operations until we either resolve all mismatches or determine it's impossible.
    
    # Convert strings to lists for easier manipulation
    S = list(S)
    T = list(T)
    
    # Helper function to check if a substring of S starting at index `start` matches the pattern of `count` characters `char`
    def check_pattern(start, count, char):
        return all(S[start + i] == char for i in range(count))
    
    # Helper function to perform the swap operation at index `start`
    def perform_swap(start):
        # Swap X zeros with Y ones or vice versa
        for i in range(X):
            S[start + i] = '1'
        for i in range(Y):
            S[start + X + i] = '0'
    
    # Iterate over the string positions
    i = 0
    while i <= N - (X + Y):
        if S[i:i+X] == T[i:i+X] and S[i+X:i+X+Y] == T[i+X:i+X+Y]:
            # Current segment matches, skip it
            i += X + Y
            continue
        
        # Check if we can apply Operation A or B to fix the current segment
        if check_pattern(i, X, '0') and check_pattern(i + X, Y, '1'):
            # This matches the pattern for Operation A, check if T requires a swap here
            if T[i:i+X] == [ '1' ] * X and T[i+X:i+X+Y] == [ '0' ] * Y:
                perform_swap(i)
                i += X + Y
                continue
        elif check_pattern(i, Y, '1') and check_pattern(i + Y, X, '0'):
            # This matches the pattern for Operation B, check if T requires a swap here
            if T[i:i+Y] == [ '0' ] * Y and T[i+Y:i+Y+X] == [ '1' ] * X:
                perform_swap(i)
                i += X + Y
                continue
        
        # If no operation can be applied, move to the next character
        i += 1
    
    # After attempting transformations, check if S matches T
    if S == T:
        return "Yes"
    else:
        return "No"

# Read input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
X = int(data[1])
Y = int(data[2])
S = data[3]
T = data[4]

# Solve the problem
result = can_transform(N, X, Y, S, T)

# Output the result
print(result)
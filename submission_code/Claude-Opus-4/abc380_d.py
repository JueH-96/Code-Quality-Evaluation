def swap_case(c):
    if c.isupper():
        return c.lower()
    else:
        return c.upper()

def find_char(S, K):
    n = len(S)
    K -= 1  # Convert to 0-indexed
    
    # Track number of case swaps
    swaps = 0
    
    # Work backwards through the operations
    while K >= n:
        # Find the length of string before this operation
        prev_len = n
        while prev_len * 2 <= K:
            prev_len *= 2
        
        # If K is in the second half of this block, we need a case swap
        if K >= prev_len:
            K -= prev_len
            swaps += 1
    
    # Get the character from original string
    char = S[K]
    
    # Apply case swaps
    if swaps % 2 == 1:
        char = swap_case(char)
    
    return char

# Read input
S = input().strip()
Q = int(input())
K_values = list(map(int, input().split()))

# Process queries
results = []
for K in K_values:
    results.append(find_char(S, K))

# Output results
print(' '.join(results))
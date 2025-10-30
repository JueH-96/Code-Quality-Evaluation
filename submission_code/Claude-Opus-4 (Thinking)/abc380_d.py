def swap_case(c):
    if c.isupper():
        return c.lower()
    else:
        return c.upper()

def find_kth_char(S, K):
    original_len = len(S)
    
    # Find the effective length we need to consider
    length = original_len
    while length < K:
        length *= 2
    
    # Now, work backwards to find the character
    swap_count = 0
    while length > original_len:
        half_length = length // 2
        if K > half_length:
            K -= half_length
            swap_count += 1
        length = half_length
    
    char = S[K - 1]  # Convert to 0-indexed
    if swap_count % 2 == 1:
        char = swap_case(char)
    
    return char

# Read input
S = input().strip()
Q = int(input().strip())
Ks = list(map(int, input().strip().split()))

# Process queries
results = []
for K in Ks:
    results.append(find_kth_char(S, K))

# Output
print(' '.join(results))
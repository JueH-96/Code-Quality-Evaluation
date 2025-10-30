# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    S = data[0]
    Q = int(data[1])
    queries = list(map(int, data[2:]))
    
    len_S = len(S)
    double_len_S = 2 * len_S
    
    # Prepare the toggled version of S
    T = ''.join(c.lower() if c.isupper() else c.upper() for c in S)
    
    # Prepare the full pattern of length 2 * len(S)
    full_pattern = S + T
    
    # Answer each query
    results = []
    for K in queries:
        # Find the effective position in the repeating pattern
        effective_position = (K - 1) % double_len_S
        results.append(full_pattern[effective_position])
    
    # Print the results
    print(' '.join(results))

main()
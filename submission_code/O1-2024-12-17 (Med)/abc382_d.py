def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data)
    
    results = []
    current = []

    def backtrack(i, prev_val):
        # If we've chosen all N elements, store the current sequence
        if i == N:
            results.append(current[:])
            return
        
        # The smallest allowed value for A_i
        start = 1 if i == 0 else prev_val + 10
        # The largest allowed value for A_i so that we still can fit the remaining elements
        # respecting the +10 condition and A_N <= M
        # M - 10*(N-1-i) gives the highest possible value for the current position
        max_val = M - 10*(N-1-i)
        
        for v in range(start, max_val + 1):
            current.append(v)
            backtrack(i + 1, v)
            current.pop()

    # Start backtracking from the first element (index 0, no previous value yet)
    backtrack(0, 0)

    # Print the number of valid sequences
    print(len(results))
    # Print each sequence in lexicographical order
    for seq in results:
        print(*seq)

# Do not forget to call main!
main()
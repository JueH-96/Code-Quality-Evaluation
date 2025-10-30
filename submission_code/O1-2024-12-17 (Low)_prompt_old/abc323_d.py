def solve():
    import sys
    from collections import defaultdict
    
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    # Dictionary to keep track of the (size -> count) after all merges
    size_counts = defaultdict(int)
    
    # Function that adds 'c' slimes of 'size' into our dictionary
    # and greedily merges them upwards.
    def add_slimes(size, c):
        while c > 0:
            size_counts[size] += c
            # Number of pairs that can be merged at this size
            merge_pairs = size_counts[size] // 2
            # Leftover slimes at this size (either 0 or 1)
            size_counts[size] %= 2
            # Move on to the next size (2 * size) with the slimes created by merging
            c = merge_pairs
            size <<= 1
    
    idx = 1
    for _ in range(N):
        s = int(input_data[idx]); idx += 1
        c = int(input_data[idx]); idx += 1
        add_slimes(s, c)
    
    # The final answer is simply the sum of leftover slimes across all sizes
    print(sum(size_counts.values()))
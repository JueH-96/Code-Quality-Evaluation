def main():
    import sys
    
    # Read the number of bottles
    N = int(sys.stdin.readline())
    
    # Compute the minimum number of friends needed (bits needed to distinguish up to N)
    M = (N - 1).bit_length()
    
    # Print the number of friends
    print(M)
    sys.stdout.flush()
    
    # For each friend i (0-based), select the bottles that have the i-th bit set in (j-1).
    for i in range(M):
        group = []
        for j in range(1, N + 1):
            if ((j - 1) >> i) & 1:
                group.append(j)
        # Print count of bottles followed by the bottle indices in ascending order
        print(len(group), *group)
        sys.stdout.flush()
    
    # Read the string that indicates which friends got sick
    S = sys.stdin.readline().strip()
    
    # Decode the spoiled bottle index using the bits
    result_index = 0
    for i, c in enumerate(S):
        if c == '1':
            result_index += (1 << i)
    
    # The actual spoiled bottle (1-based)
    print(result_index + 1)
    sys.stdout.flush()

# Do not forget to call main()
main()
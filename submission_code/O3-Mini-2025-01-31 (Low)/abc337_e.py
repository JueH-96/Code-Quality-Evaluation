def main():
    import sys
    input = sys.stdin.readline
    
    # Read N
    N = int(input().strip())
    
    # Determine number of friends M = ceil(log2(N))
    M = 0
    tmp = N
    while (1 << M) < N:
        M += 1
    
    # Print the number of friends M
    sys.stdout.write(str(M) + "
")
    sys.stdout.flush()
    
    # For each friend, determine the bottles to serve.
    # We use the binary representation of (bottle_number-1) for easier bit extraction.
    # For friend i (0-indexed), they receive bottle j if the i-th bit of (j-1) is 1.
    for friend in range(M):
        bottles = []
        # iterate over bottle numbers 1 to N.
        for bottle in range(1, N+1):
            if (bottle - 1) & (1 << friend):
                bottles.append(bottle)
        # Print count and list. The problem requires the list to be in ascending order.
        # Since we are iterating in order, they are naturally in ascending order.
        out_line = str(len(bottles))
        for b in bottles:
            out_line += " " + str(b)
        sys.stdout.write(out_line + "
")
        sys.stdout.flush()
    
    # Now read the string of results S from the judge.
    S = input().strip()
    
    # Reconstruct the bit pattern from S. For friend i (0-indexed), if S[i]=='1'
    # then bit i in the binary representation is 1.
    res = 0
    for i in range(M):
        if S[i] == '1':
            res |= (1 << i)
            
    # Because we used binary representation of (bottle-1), we add 1.
    spoiled = res + 1
    sys.stdout.write(str(spoiled) + "
")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
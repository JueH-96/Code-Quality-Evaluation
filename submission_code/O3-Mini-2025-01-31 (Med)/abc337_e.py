def main():
    import sys
    import math

    input = sys.stdin.readline
    
    # Read the number of bottles N.
    N_line = input().strip()
    if not N_line:
        return
    N = int(N_line)
    
    # The minimum number of friends is ceil(log2(N)).
    M = math.ceil(math.log2(N))
    
    # Print the number of friends.
    print(M)
    sys.stdout.flush()
    
    # For each friend i from 0 to M-1, choose all bottles j for which bit i is set in (j-1).
    for i in range(M):
        bottles = []
        for j in range(1, N + 1):
            # If bit i of (j-1) is 1, then friend i tests bottle j.
            if ((j - 1) >> i) & 1:
                bottles.append(j)
        
        # Prepare the output string.
        if bottles:
            line = "{} {}".format(len(bottles), " ".join(map(str, bottles)))
        else:
            line = "0"
        print(line)
        sys.stdout.flush()
    
    # Read the judge's response: an M-length string indicating friend results.
    S = input().strip()
    
    # Reconstruct the spoiled bottle number:
    # Each friend corresponds to bit i in the (j - 1) binary representation.
    bad_bottle = 1
    for i in range(M):
        if S[i] == '1':
            bad_bottle += (1 << i)
    
    # Output the identified bad bottle number.
    print(bad_bottle)
    sys.stdout.flush()

if __name__ == '__main__':
    main()
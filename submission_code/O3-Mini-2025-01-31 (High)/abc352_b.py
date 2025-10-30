def main():
    import sys
    input_data = sys.stdin.read().split()
    
    # Read the strings S and T
    S = input_data[0].strip()
    T = input_data[1].strip()
    
    # Use two pointers: i for S and j for T.
    i = 0
    j = 0
    ans = []
    
    # We loop through T, matching characters in order with S.
    while i < len(S) and j < len(T):
        if S[i] == T[j]:
            # If they match, record (j+1) (1-indexed) and move to the next letter in S.
            ans.append(str(j + 1))
            i += 1
        # Whether match or not, move to the next letter in T.
        j += 1
    
    # Output the recorded positions, separated by spaces.
    sys.stdout.write(" ".join(ans))
    
if __name__ == '__main__':
    main()
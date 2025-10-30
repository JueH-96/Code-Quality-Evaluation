def solve():
    S, T = input().split()
    N = len(S)

    # Iterate over possible values for w
    # w must be less than |S|, so w ranges from 1 to N-1.
    # range(1, N) gives 1, 2, ..., N-1.
    for w in range(1, N):
        # Iterate over possible values for c
        # c must be between 1 and w (inclusive), i.e., 1 <= c <= w.
        # range(1, w + 1) gives 1, 2, ..., w.
        for c in range(1, w + 1):
            char_list = []  # To store characters for the constructed string
            
            # Iterate through S in blocks of size w
            # start_idx is the 0-based starting index of the current block
            for start_idx in range(0, N, w):
                # Extract the substring for the current block
                # S[start_idx : start_idx + w] will correctly slice S.
                # If start_idx + w > N, it slices up to the end of S.
                substring = S[start_idx : start_idx + w]
                
                # Condition: substring length must be at least c
                # c is 1-indexed as per problem statement.
                if len(substring) >= c:
                    # Append the c-th character (which is at index c-1 in 0-based Python)
                    char_list.append(substring[c-1])
            
            constructed_string = "".join(char_list)
            
            # Check if the constructed string matches T
            if constructed_string == T:
                print("Yes")
                return  # Found a valid pair, so exit
    
    # If loops complete, no valid (c, w) pair was found
    print("No")

if __name__ == '__main__':
    solve()
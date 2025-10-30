# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Find all 1-blocks
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1':
            l = i
            while i < N and S[i] == '1':
                i += 1
            r = i - 1
            blocks.append((l, r))
        else:
            i += 1
    
    # Get the (K-1)-th and K-th 1-blocks
    l_k_minus_1, r_k_minus_1 = blocks[K-2]
    l_k, r_k = blocks[K-1]
    
    # Construct the new string T
    # Part before the (K-1)-th block remains the same
    T = list(S[:r_k_minus_1 + 1])
    
    # Insert the K-th block after the (K-1)-th block
    T.extend(['1'] * (r_k - l_k + 1))
    
    # Insert the zeros that were between the (K-1)-th and K-th blocks
    # The zeros are from r_k_minus_1 + 1 to l_k - 1
    T.extend(['0'] * (l_k - (r_k_minus_1 + 1)))
    
    # Append the rest of the string after the K-th block
    T.extend(list(S[r_k + 1:]))
    
    # Convert list to string
    T = ''.join(T)
    
    print(T)

if __name__ == "__main__":
    main()
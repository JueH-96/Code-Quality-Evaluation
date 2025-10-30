def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, K = map(int, data[:2])
    S = data[2]
    
    # 1) Identify all 1-blocks (start, end) in 0-based indexing.
    blocks = []
    i = 0
    while i < N:
        if S[i] == '1':
            start = i
            while i < N and S[i] == '1':
                i += 1
            end = i - 1
            blocks.append((start, end))
        else:
            i += 1
    
    # 2) Get the (K-1)-th block and K-th block in 1-based sense
    #    but they are blocks[K-2] and blocks[K-1] in 0-based indexing.
    l_km1, r_km1 = blocks[K - 2]  # (K-1)-th block
    l_k, r_k = blocks[K - 1]      # K-th block
    
    # 3) According to the problem statement, construct T:
    #    - T[:r_km1+1]  = S[:r_km1+1]
    #    - Next (r_k - l_k + 1) positions = '1'
    #    - Until index r_k = '0'
    #    - T[r_k+1:] = S[r_k+1:]
    T = list(S)  # We will overwrite the necessary parts below.
    
    # For clarity, rebuild T from scratch to match exactly the problem's formula:
    T = [None] * N
    
    # Copy S up to r_km1
    for i in range(r_km1 + 1):
        T[i] = S[i]
    
    block_len = r_k - l_k + 1
    
    # Fill with '1' for the length of K-th block
    # from r_km1+1 to r_km1 + block_len (inclusive)
    for i in range(r_km1 + 1, r_km1 + 1 + block_len):
        T[i] = '1'
    
    # Fill with '0' from r_km1 + block_len + 1 to r_k (inclusive)
    for i in range(r_km1 + 1 + block_len, r_k + 1):
        T[i] = '0'
    
    # Copy the remainder from S
    for i in range(r_k + 1, N):
        T[i] = S[i]
    
    # 4) Print the result
    print("".join(T))


# Do not forget to call main()
if __name__ == "__main__":
    main()
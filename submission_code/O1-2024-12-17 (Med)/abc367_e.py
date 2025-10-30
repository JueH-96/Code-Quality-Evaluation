def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Fast extraction of inputs
    N = int(input_data[0])
    K = int(input_data[1])
    X = list(map(int, input_data[2:2+N]))
    A = list(map(int, input_data[2+N:2+2*N]))
    
    # If K = 0, we can simply output A and return
    if K == 0:
        print(*A)
        return
    
    # Build 0-based "next" array: next_pos[i] = X[i] - 1
    next_pos = [x-1 for x in X]
    
    # We will use a "doubling" (binary lifting) approach to skip up to K steps.
    # max_d is enough bits to cover K
    max_d = K.bit_length()  # up to ~60 if K ~1e18
    
    # next_table[d][i] will tell us where index i goes after 2^d applications
    # of the transformation "i -> next_pos[i]".
    next_table = [next_pos]  # layer 0: 1 step
    
    # Precompute all layers
    for d in range(1, max_d):
        prev_layer = next_table[d-1]
        curr_layer = [0]*N
        for i in range(N):
            curr_layer[i] = prev_layer[prev_layer[i]]
        next_table.append(curr_layer)
    
    # For each i, we want to find its position after K applications.
    # Then the final value in A is at that position.
    ans = [0]*N
    
    for i in range(N):
        pos = i
        steps = K
        bit = 0
        while steps > 0:
            if steps & 1:
                pos = next_table[bit][pos]
            steps >>= 1
            bit += 1
        ans[i] = A[pos]
    
    print(*ans)

# Call main() to adhere to the requirement
if __name__ == "__main__":
    main()
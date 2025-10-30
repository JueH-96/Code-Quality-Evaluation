def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1]
    
    # Precompute the number of consecutive '1's ending at each position
    left_ones = [0]*N
    if S[0] == '1':
        left_ones[0] = 1
    for i in range(1, N):
        if S[i] == '1':
            left_ones[i] = left_ones[i-1] + 1
        else:
            left_ones[i] = 0
    
    # Precompute the number of consecutive '2's starting at each position
    right_twos = [0]*N
    if S[N-1] == '2':
        right_twos[N-1] = 1
    for i in range(N-2, -1, -1):
        if S[i] == '2':
            right_twos[i] = right_twos[i+1] + 1
        else:
            right_twos[i] = 0
    
    # We know there's at least one slash, so the minimum answer can be 1
    # (the "/" alone, which is a valid 11/22 string of length 1).
    ans = 1
    
    # Check each position for '/', and compute the maximum possible 11/22 substring
    for i in range(N):
        if S[i] == '/':
            left_count = left_ones[i-1] if i > 0 else 0
            right_count = right_twos[i+1] if i < N-1 else 0
            # The substring can be 1^k / 2^k with k = min(left_count, right_count)
            length = 2 * min(left_count, right_count) + 1
            ans = max(ans, length)
    
    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()
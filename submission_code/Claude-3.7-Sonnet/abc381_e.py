def main():
    # Read input
    N, Q = map(int, input().split())
    S = input()
    
    # Precompute prefix count of '1's and suffix count of '2's
    count_1_prefix = [0] * (N + 1)
    count_2_suffix = [0] * (N + 1)
    
    for i in range(N):
        count_1_prefix[i+1] = count_1_prefix[i] + (1 if S[i] == '1' else 0)
    
    for i in range(N-1, -1, -1):
        count_2_suffix[i] = count_2_suffix[i+1] + (1 if S[i] == '2' else 0)
    
    # Process each query
    for _ in range(Q):
        L, R = map(int, input().split())
        L -= 1  # Convert to 0-indexed
        R -= 1
        
        max_length = 0
        # For each '/' in the substring, calculate the longest possible 11/22 string
        for idx in range(L, R+1):
            if S[idx] == '/':
                # Count '1's before this '/' and '2's after this '/'
                count_1 = count_1_prefix[idx] - count_1_prefix[L]
                count_2 = count_2_suffix[idx+1] - count_2_suffix[R+1]
                
                # Length of 11/22 string = 2*min(count_1, count_2) + 1
                length = 2 * min(count_1, count_2) + 1
                max_length = max(max_length, length)
        
        print(max_length)

if __name__ == "__main__":
    main()
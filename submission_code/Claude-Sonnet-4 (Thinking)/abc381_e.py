N, Q = map(int, input().split())
S = input().strip()

# Precompute prefix sums
prefix_ones = [0] * (N + 1)
prefix_twos = [0] * (N + 1)

for i in range(N):
    prefix_ones[i+1] = prefix_ones[i] + (1 if S[i] == '1' else 0)
    prefix_twos[i+1] = prefix_twos[i] + (1 if S[i] == '2' else 0)

for _ in range(Q):
    L, R = map(int, input().split())
    # Convert to 0-indexed
    L -= 1
    R -= 1
    
    max_length = 0
    
    # Try each '/' as the middle character
    for i in range(L, R+1):
        if S[i] == '/':
            # Count '1's to the left
            left_ones = prefix_ones[i] - prefix_ones[L]
            
            # Count '2's to the right
            right_twos = prefix_twos[R+1] - prefix_twos[i+1]
            
            # Calculate max length for this '/'
            length = 2 * min(left_ones, right_twos) + 1
            max_length = max(max_length, length)
    
    print(max_length)
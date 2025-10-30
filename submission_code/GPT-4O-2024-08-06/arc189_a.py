# YOUR CODE HERE
def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # Initial state based on i % 2
    initial = [(i % 2) for i in range(1, N + 1)]
    
    # Find segments in A
    segments = []
    current_value = A[0]
    current_length = 1
    
    for i in range(1, N):
        if A[i] == current_value:
            current_length += 1
        else:
            segments.append((current_value, current_length))
            current_value = A[i]
            current_length = 1
    
    # Append the last segment
    segments.append((current_value, current_length))
    
    # Calculate the number of ways to achieve each segment
    result = 1
    for value, length in segments:
        # Determine the starting index parity
        if value == 1:
            # Must start from an odd index
            start_parity = 1
        else:
            # Must start from an even index
            start_parity = 0
        
        # Calculate the number of valid starting positions
        # We need to find how many positions of the correct parity are there
        # in the range of the segment length
        valid_starts = (length + 1) // 2
        
        # Multiply the result by the number of valid starting positions
        result = (result * valid_starts) % MOD
    
    print(result)
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Pair each element with its original position (1-based)
    indexed_A = [(A[i], i+1) for i in range(N)]
    # Sort based on the element values
    indexed_A.sort(key=lambda x: x[0])
    
    # Precompute an array to mark where positions are strictly increasing
    valid = [1] * (N-1)
    for i in range(N-1):
        if indexed_A[i+1][1] <= indexed_A[i][1]:
            valid[i] = 0
    
    # Create a prefix sum array for the 'valid' array
    prefix = [0] * (N)
    for i in range(N-1):
        prefix[i+1] = prefix[i] + valid[i]
    
    min_diff = float('inf')
    window_size = N - K
    for i in range(N - window_size + 1):
        # Check if all pairs in this window are strictly increasing in position
        start = i
        end = i + window_size - 1
        # Number of valid pairs in this window
        num_valid = prefix[end] - prefix[start]
        # Total possible pairs in the window
        total_pairs = window_size - 1
        if num_valid == total_pairs:
            # All pairs are valid, calculate the difference
            current_diff = indexed_A[i + window_size - 1][0] - indexed_A[i][0]
            if current_diff < min_diff:
                min_diff = current_diff
    
    print(min_diff)

if __name__ == '__main__':
    main()
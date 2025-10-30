def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    if K == 1:
        print(0)
        return
    
    # Create a dictionary to map each value to its index
    value_to_index = {value: idx for idx, value in enumerate(P)}
    
    min_diff = float('inf')
    
    # Iterate over all possible starting values 'a' for the consecutive sequence
    for a in range(1, N - K + 2):
        # Get the indices of the values a, a+1, ..., a+K-1
        indices = [value_to_index.get(val, -1) for val in range(a, a + K)]
        # Check if all values are present
        if -1 in indices:
            continue
        # Sort the indices to find the smallest window
        indices.sort()
        # Calculate the difference between the last and first index
        current_diff = indices[-1] - indices[0]
        if current_diff < min_diff:
            min_diff = current_diff
    
    print(min_diff)

if __name__ == "__main__":
    main()
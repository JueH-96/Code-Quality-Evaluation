def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    B = list(map(int, data[2:2+N]))
    W = list(map(int, data[2+N:2+N+M]))
    
    # Sort black and white balls by value in descending order
    B.sort(reverse=True)
    W.sort(reverse=True)
    
    # Calculate prefix sums for black and white balls
    prefix_black = [0] * (N + 1)
    prefix_white = [0] * (M + 1)
    
    for i in range(1, N+1):
        prefix_black[i] = prefix_black[i-1] + B[i-1]
    
    for i in range(1, M+1):
        prefix_white[i] = prefix_white[i-1] + W[i-1]
    
    # We can always choose to take no balls
    max_sum = 0
    
    # Try to take k white balls, and at least k black balls
    for k in range(min(N, M) + 1):
        if k <= M:
            current_sum = prefix_white[k]
        else:
            current_sum = 0
        
        if k <= N:
            current_sum += prefix_black[k]
        
        max_sum = max(max_sum, current_sum)
    
    # Output the maximum sum found
    print(max_sum)

if __name__ == "__main__":
    main()
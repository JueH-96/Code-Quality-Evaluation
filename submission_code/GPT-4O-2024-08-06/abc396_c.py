# YOUR CODE HERE
def max_sum_of_balls():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    B = list(map(int, data[2:2+N]))
    W = list(map(int, data[2+N:2+N+M]))
    
    # Sort black balls in descending order of value
    B.sort(reverse=True)
    # Sort white balls in descending order of value
    W.sort(reverse=True)
    
    # Calculate prefix sums for black balls
    prefix_black = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_black[i] = prefix_black[i - 1] + B[i - 1]
    
    # Calculate prefix sums for white balls
    prefix_white = [0] * (M + 1)
    for j in range(1, M + 1):
        prefix_white[j] = prefix_white[j - 1] + W[j - 1]
    
    # We need to find the maximum sum where number of black balls >= number of white balls
    max_sum = 0
    
    # Try choosing k white balls
    for k in range(M + 1):
        # We need at least k black balls
        if k <= N:
            # Sum of top k white balls
            sum_white = prefix_white[k]
            # Sum of top k black balls
            sum_black = prefix_black[k]
            # Total sum
            total_sum = sum_white + sum_black
            # Update max_sum
            max_sum = max(max_sum, total_sum)
    
    print(max_sum)
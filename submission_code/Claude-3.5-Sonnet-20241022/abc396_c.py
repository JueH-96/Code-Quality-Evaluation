def solve():
    # Read input
    N, M = map(int, input().split())
    black = list(map(int, input().split()))
    white = list(map(int, input().split()))
    
    # Sort both arrays in descending order
    black.sort(reverse=True)
    white.sort(reverse=True)
    
    # Initialize prefix sums for both arrays
    black_prefix = [0] * (N + 1)
    white_prefix = [0] * (M + 1)
    
    for i in range(N):
        black_prefix[i + 1] = black_prefix[i] + black[i]
    for i in range(M):
        white_prefix[i + 1] = white_prefix[i] + white[i]
    
    # Initialize answer as 0 (case when we choose no balls)
    answer = 0
    
    # Try all possible numbers of white balls
    for white_count in range(M + 1):
        # We need at least as many black balls as white balls
        min_black = white_count
        
        # For each number of white balls, try all valid numbers of black balls
        for black_count in range(min_black, N + 1):
            # Calculate sum for this combination
            current_sum = black_prefix[black_count] + white_prefix[white_count]
            answer = max(answer, current_sum)
    
    print(answer)

# Run the solution
solve()
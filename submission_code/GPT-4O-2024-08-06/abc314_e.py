def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    index += 1
    M = int(data[index])
    index += 1
    
    wheels = []
    
    for _ in range(N):
        C = int(data[index])
        index += 1
        P = int(data[index])
        index += 1
        S = list(map(int, data[index:index + P]))
        index += P
        wheels.append((C, P, S))
    
    # Initialize dp array
    dp = [float('inf')] * (M + 1)
    dp[0] = 0.0
    
    # Process each wheel
    for C, P, S in wheels:
        expected_points = sum(S) / P
        expected_cost = C / expected_points
        
        # Update dp array
        for current_points in range(M, -1, -1):
            for points in S:
                new_points = min(M, current_points + points)
                dp[new_points] = min(dp[new_points], dp[current_points] + expected_cost)
    
    # The answer is the expected cost to reach at least M points
    print(dp[M])
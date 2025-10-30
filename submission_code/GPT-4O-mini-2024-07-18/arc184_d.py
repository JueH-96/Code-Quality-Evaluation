def count_remaining_ball_sets(N, balls):
    MOD = 998244353
    
    # Sort balls based on their X and Y coordinates
    balls.sort()
    
    # dp[i] will store the number of valid sets that can be formed with the first i balls
    dp = [0] * (N + 1)
    dp[0] = 1  # Base case: one way to choose nothing
    
    # We will use a list to keep track of the last valid position
    last_valid = [0] * (N + 1)
    
    for i in range(1, N + 1):
        last_valid[i] = i  # Initially, all are valid
        for j in range(1, i):
            # Check if we can remove ball j when considering ball i
            if (balls[j - 1][0] < balls[i - 1][0] and balls[j - 1][1] < balls[i - 1][1]) or \
               (balls[j - 1][0] > balls[i - 1][0] and balls[j - 1][1] > balls[i - 1][1]):
                last_valid[i] = j  # Update the last valid position
        
        # Update dp[i] based on the last valid position
        dp[i] = (dp[i - 1] + dp[last_valid[i] - 1]) % MOD
    
    # The answer is the sum of all dp[i] for i from 1 to N
    result = sum(dp[1:N + 1]) % MOD
    return result

import sys
input = sys.stdin.read

def main():
    data = input().strip().splitlines()
    N = int(data[0])
    balls = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = count_remaining_ball_sets(N, balls)
    print(result)

if __name__ == "__main__":
    main()
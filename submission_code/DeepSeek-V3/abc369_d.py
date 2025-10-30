def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Initialize DP table
    # dp[i][j] where i is the index of the monster, j is the count of defeated monsters so far
    # Since N can be up to 2e5, we need a more efficient way
    # Instead, we can manage with two variables: one for even count, one for odd count
    
    # Initialize:
    # dp_odd: when the count of defeated monsters is odd
    # dp_even: when the count of defeated monsters is even
    dp_odd = 0
    dp_even = 0
    
    for i in range(N):
        current_A = A[i]
        # Calculate new dp_odd and dp_even
        # If we defeat the current monster:
        # If the previous count was even, now it's odd, and we get current_A * 1
        # If the previous count was odd, now it's even, and we get current_A * 2
        new_dp_odd = max(dp_odd, dp_even + current_A)
        new_dp_even = max(dp_even, dp_odd + 2 * current_A)
        
        # Also, if we don't defeat the current monster, the count remains the same
        new_dp_odd = max(new_dp_odd, dp_odd)
        new_dp_even = max(new_dp_even, dp_even)
        
        dp_odd, dp_even = new_dp_odd, new_dp_even
    
    # The maximum is the maximum of dp_odd and dp_even
    print(max(dp_odd, dp_even))

if __name__ == "__main__":
    main()
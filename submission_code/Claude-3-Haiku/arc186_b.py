# Read the input
N = int(input())
A = list(map(int, input().split()))

# Define a function to calculate the number of permutations
def count_permutations(N, A):
    MOD = 998244353
    
    # Create a list to store the number of permutations for each index
    dp = [0] * N
    
    # Initialize the first element
    dp[0] = 1
    
    for i in range(1, N):
        # Count the number of valid permutations for the current index
        for j in range(A[i], i):
            dp[i] = (dp[i] + dp[j]) % MOD
        
        # If A[i] is not 0, we need to ensure that P[A[i]] < P[i]
        if A[i] > 0:
            dp[i] = (dp[i] - dp[A[i]]) % MOD
            if dp[i] < 0:
                dp[i] += MOD
    
    # Return the total number of permutations
    return sum(dp) % MOD

# Print the result
print(count_permutations(N, A))
# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    MOD = 998244353
    
    N = int(data[0])
    Q = int(data[1])
    
    P = []
    V = []
    index = 2
    for _ in range(Q):
        P.append(int(data[index]))
        V.append(int(data[index + 1]))
        index += 2
    
    # dp[i][0] means the number of ways to perform the first i operations
    # with the i-th operation choosing the first option
    # dp[i][1] means the number of ways to perform the first i operations
    # with the i-th operation choosing the second option
    dp = [[0, 0] for _ in range(Q)]
    
    # Initialize the first operation
    dp[0][0] = 1
    dp[0][1] = 1
    
    # Iterate over each operation
    for i in range(1, Q):
        # If we choose the first option for operation i
        # We need to ensure that all elements from 1 to P[i] are <= V[i]
        # This means the previous operation must have set these elements to <= V[i]
        if V[i] >= V[i-1]:
            dp[i][0] = (dp[i-1][0] + dp[i-1][1]) % MOD
        
        # If we choose the second option for operation i
        # We need to ensure that all elements from P[i] to N are <= V[i]
        # This means the previous operation must have set these elements to <= V[i]
        if V[i] >= V[i-1]:
            dp[i][1] = (dp[i-1][0] + dp[i-1][1]) % MOD
    
    # The answer is the sum of the ways to perform all Q operations
    result = (dp[Q-1][0] + dp[Q-1][1]) % MOD
    print(result)

main()
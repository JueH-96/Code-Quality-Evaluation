def solve_query3(A, B, l, r):
    # dp[i][j] represents max value at position i
    # j=0: last operation was addition, j=1: last operation was multiplication
    dp = [[0, 0] for _ in range(r-l+2)]
    
    # Base case
    dp[0][0] = 0  # for addition
    dp[0][1] = 0  # for multiplication
    
    # For each position
    for i in range(l-1, r):
        pos = i - (l-1) + 1
        # Try addition
        add_from_add = dp[pos-1][0] + A[i]
        add_from_mul = dp[pos-1][1] + A[i]
        
        # Try multiplication
        mul_from_add = dp[pos-1][0] * B[i]
        mul_from_mul = dp[pos-1][1] * B[i]
        
        # Take maximum for each case
        dp[pos][0] = max(add_from_add, add_from_mul)
        dp[pos][1] = max(mul_from_add, mul_from_mul)
    
    return max(dp[r-l+1])

# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))
Q = int(input())

# Process queries
for _ in range(Q):
    query = list(map(int, input().split()))
    
    if query[0] == 1:
        # Update A
        _, i, x = query
        A[i-1] = x
    elif query[0] == 2:
        # Update B
        _, i, x = query
        B[i-1] = x
    else:
        # Type 3 query
        _, l, r = query
        result = solve_query3(A, B, l, r)
        print(result)
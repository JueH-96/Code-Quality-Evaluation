# YOUR CODE HERE
import sys
sys.setrecursionlimit(10**6)

def solve():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    N = int(data[0])
    A = int(data[1])
    X = int(data[2])
    Y = int(data[3])
    
    dp = {}
    
    def expected_cost(n):
        if n == 0:
            return 0
        if n in dp:
            return dp[n]
        
        # Cost using operation A
        costA = X + expected_cost(n // A)
        
        # Cost using operation B
        costB = Y
        for b in range(1, 7):
            costB += (1/6) * expected_cost(n // b)
        
        # Minimum of both operations
        dp[n] = min(costA, costB)
        return dp[n]
    
    result = expected_cost(N)
    print(f"{result:.12f}")
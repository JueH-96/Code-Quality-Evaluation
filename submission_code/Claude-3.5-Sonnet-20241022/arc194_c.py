def calculate_cost(A, C):
    return sum(a * c for a, c in zip(A, C))

def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    C = list(map(int, input().split()))
    
    # Convert lists to arrays for easier manipulation
    A = A.copy()  # Make a copy to not modify original
    
    # For each position where A and B differ, we need to try both:
    # 1. Flipping this position now
    # 2. Not flipping this position now
    # We can use dynamic programming to solve this
    
    # dp[pos][state] represents minimum cost to match B[pos:] 
    # when current array sum costs are in state
    dp = {}
    
    def solve_dp(pos, curr_A):
        if pos == N:
            return 0
            
        state = (pos, tuple(curr_A))
        if state in dp:
            return dp[state]
            
        # If current position matches B, we can keep it as is
        if curr_A[pos] == B[pos]:
            result = solve_dp(pos + 1, curr_A)
        else:
            # We must flip this position
            # Make a copy of current array
            new_A = curr_A.copy()
            new_A[pos] = 1 - new_A[pos]
            # Cost is current sum of A * C after flip
            cost = calculate_cost(new_A, C)
            result = cost + solve_dp(pos + 1, new_A)
            
        dp[state] = result
        return result
    
    # Get minimum cost
    min_cost = solve_dp(0, A)
    print(min_cost)

solve()
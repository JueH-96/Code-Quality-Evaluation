def solve():
    # Read input
    N = int(input())
    A = list(map(int, input().split()))
    
    # dp[l][r] represents the maximum score possible for subarray A[l:r+1]
    dp = {}
    
    def get_max_score(left, right):
        # Base cases
        if right - left <= 0:  # Empty or single element
            return 0
        if right - left == 1:  # Two elements
            return abs(A[left] - A[right])
            
        # Check if already computed
        if (left, right) in dp:
            return dp[(left, right)]
            
        # Try all possible pairs of adjacent elements
        max_score = 0
        for i in range(left, right):
            # Score from current pair
            curr_score = abs(A[i] - A[i+1])
            
            # After removing pair i,i+1, we get two subarrays:
            # left to i-1 and i+2 to right
            # These become adjacent after removing i,i+1
            remaining_score = get_max_score(left, i-1) + get_max_score(i+2, right)
            
            max_score = max(max_score, curr_score + remaining_score)
            
        dp[(left, right)] = max_score
        return max_score
    
    # Get result for entire array
    result = get_max_score(0, N-1)
    print(result)

# Run the solution
solve()
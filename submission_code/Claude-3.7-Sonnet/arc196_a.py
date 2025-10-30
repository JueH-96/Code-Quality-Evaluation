def solve(A, N):
    # Memoization dictionary
    memo = {}
    
    def dp(seq):
        """
        Recursive function to compute the maximum score.
        
        Args:
            seq: A tuple representing the current sequence.
            
        Returns:
            The maximum score obtainable from the current sequence.
        """
        # Base case: if there are 0 or 1 elements, score is 0
        if len(seq) <= 1:
            return 0
        
        # Check if we've already solved this subproblem
        if seq in memo:
            return memo[seq]
        
        max_val = 0
        # Try removing each adjacent pair
        for i in range(len(seq) - 1):
            # Calculate score for removing the pair (seq[i], seq[i+1])
            # Then recursively solve for the remaining sequence
            score = abs(seq[i] - seq[i+1]) + dp(seq[:i] + seq[i+2:])
            max_val = max(max_val, score)
        
        # Memoize the result
        memo[seq] = max_val
        return max_val
    
    return dp(tuple(A))

# Reading the inputs
N = int(input())
A = list(map(int, input().split()))

# Output the result
print(solve(A, N))
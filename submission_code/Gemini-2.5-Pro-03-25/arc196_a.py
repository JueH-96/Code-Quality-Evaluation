# YOUR CODE HERE
import sys

# Increase recursion depth limit for deep recursion required by the DP.
# Set it large enough for N up to 3*10^5.
# A fixed large value is used, might need adjustment based on environment.
try:
    # Setting to N+10 (e.g. 300010) as a safe margin.
    sys.setrecursionlimit(300010) 
except Exception as e:
    # If setting limit fails, print error to stderr for debugging, but continue.
    # The program might still work if the default limit is sufficient or 
    # the environment handles deep recursion differently (e.g., PyPy).
    # print(f"Could not set recursion depth: {e}", file=sys.stderr)
    pass

# Dictionary for memoization to store results of subproblems.
# Key: tuple (i, j) representing the subarray A[i...j]
# Value: maximum score obtainable from this subarray
memo = {}

def solve(i, j, A):
    """
    Computes the maximum score for the subarray A[i...j].
    This function assumes that the length of the subarray, j-i+1, is even,
    as it represents a segment where all elements must be paired up and removed.
    Uses memoization (dynamic programming) to avoid recomputing results for the same subproblems.
    A is a 1-indexed list/array.
    """
    
    # Base case: If the interval is empty (i > j), the score is 0.
    # This occurs when a subproblem covers zero elements.
    if i > j: 
        return 0
    
    # Base case: interval of length 2 (A[i], A[i+1]). 
    # The only possible operation is to remove this pair. Score is |A[i] - A[i+1]|.
    # Here j = i + 1. This check can optimize by cutting off recursion early.
    if i == j - 1:
        return abs(A[i] - A[j])

    # Check if the result for state (i, j) is already computed and stored in memo.
    state = (i, j)
    if state in memo:
        return memo[state]

    # Initialize max_score for this state (i, j) to 0. Since scores are non-negative, 0 is a safe minimum.
    max_score = 0
    
    # The core DP recurrence logic:
    # To compute the max score for A[i...j], we must pair A[i] with some A[k].
    # For A[i] and A[k] to become adjacent and be removed together, all elements
    # in between (A[i+1...k-1]) must first be removed in pairs among themselves.
    # This requires the length of A[i+1...k-1], which is (k-1)-(i+1)+1 = k-i-1, to be even.
    # Consequently, k-i must be odd.
    # The loop iterates through all valid k: from i+1 up to j, with a step of 2.
    # This ensures that k > i and k-i is always odd.
    for k in range(i + 1, j + 1, 2): 
        
        # Recursively compute the max score for the inner segment A[i+1...k-1].
        # Length k-i-1 is even.
        score1 = solve(i + 1, k - 1, A)
        
        # Compute the score obtained from removing the pair (A[i], A[k]) once they become adjacent.
        current_pair_score = abs(A[i] - A[k])
        
        # After removing A[i...k], the remaining elements are A[k+1...j].
        # These must also be removed in pairs among themselves.
        # Recursively compute the max score for this outer segment A[k+1...j].
        # Length j-(k+1)+1 = j-k must be even (verified because j-i+1 is even and k-i-1 is even).
        score2 = solve(k + 1, j, A)
        
        # The total score for the choice of pairing A[i] with A[k] is the sum of these three parts.
        total_score_k = score1 + current_pair_score + score2
        
        # Update the maximum score found so far for the state (i, j).
        max_score = max(max_score, total_score_k)

    # Store the computed maximum score in the memoization table before returning.
    memo[state] = max_score
    return max_score

def main():
    # Read input N
    N = int(sys.stdin.readline())
    # Read sequence A as a list of integers
    A_input = list(map(int, sys.stdin.readline().split()))

    # Use a 1-based indexing for the array A internally to align with DP state definitions (i, j).
    # A[0] is a dummy placeholder value (0 is fine), actual elements are A[1]...A[N].
    A = [0] + A_input 

    if N % 2 == 0:
        # If N is even, all elements must be removed in pairs.
        # The problem asks for the maximum score for the entire sequence A[1...N].
        # The length N is even, so we can directly call solve(1, N).
        result = solve(1, N, A)
        print(result)
    else:
        # If N is odd, exactly one element will remain after all operations.
        # This remaining element must have an odd original index (1, 3, 5, ...).
        # We need to find which choice of remaining element k yields the maximum total score.
        # The total score is obtained by removing pairs from the segments left and right of k.
        
        max_total_score = 0
        # Iterate through all possible odd indices k for the element that could remain.
        for k in range(1, N + 1, 2):
            # Calculate the score obtained from removing pairs in the left segment A[1...k-1].
            # The length k-1 is even because k is odd.
            score_left = solve(1, k - 1, A)
            
            # Calculate the score obtained from removing pairs in the right segment A[k+1...N].
            # The length N-k is even because N and k are both odd.
            score_right = solve(k + 1, N, A)
            
            # The total score for leaving element k is the sum of scores from the left and right segments.
            total_score = score_left + score_right
            
            # Keep track of the maximum total score found across all possible choices of k.
            max_total_score = max(max_total_score, total_score)
            
        print(max_total_score)

# Standard check to run the main function when the script is executed.
if __name__ == '__main__':
    main()
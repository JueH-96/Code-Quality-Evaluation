def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    total_votes = sum(A)
    remaining_votes = K - total_votes
    
    # Create a list of tuples (A_i, index) to sort while keeping track of original indices
    indexed_A = [(A[i], i) for i in range(N)]
    # Sort in descending order of A_i
    indexed_A.sort(reverse=True, key=lambda x: x[0])
    
    # Initialize the result array
    result = [0] * N
    
    for i in range(N):
        original_index = indexed_A[i][1]
        current_votes = indexed_A[i][0]
        
        # Determine the M-th candidate's votes after adding X to current_votes
        # We need to ensure that at most M-1 candidates have more votes than current_votes + X
        
        # The M-th candidate's votes after adding X should be <= current_votes + X
        # So, the (M-1)-th candidate's votes should be <= current_votes + X
        
        # Find the (M-1)-th candidate's current votes
        if M-1 < N:
            mth_votes = indexed_A[M-1][0]
        else:
            mth_votes = 0
        
        # To ensure that current_votes + X >= mth_votes
        # So, X >= mth_votes - current_votes
        # But also, X must be <= remaining_votes
        
        required_X = max(0, mth_votes - current_votes)
        
        if required_X > remaining_votes:
            result[original_index] = -1
        else:
            result[original_index] = required_X
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()
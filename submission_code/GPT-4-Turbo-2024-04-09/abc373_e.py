def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    K = int(data[2])
    A = list(map(int, data[3:3+N]))
    
    total_votes = sum(A)
    remaining_votes = K - total_votes
    
    # Sort candidates by their current votes
    sorted_indices = sorted(range(N), key=lambda i: A[i])
    
    # This will store the minimum additional votes needed for each candidate
    min_additional_votes = [0] * N
    
    # To determine the minimum votes needed for each candidate to be in the top N-M positions
    for idx in range(N):
        candidate_idx = sorted_indices[idx]
        current_votes = A[candidate_idx]
        
        # We need this candidate to have more votes than at least N-M other candidates
        # To be safe, we want to be better than the M-th candidate from the top
        # which is the (N-M)th candidate in 0-indexed sorted list
        if idx < N - M:
            # Candidate cannot be in the top N-M even if they get all remaining votes
            min_additional_votes[candidate_idx] = -1
        else:
            # Calculate the minimum votes needed to be better than the (N-M-1)-th candidate
            # if M == N, then we just need to be better than the worst candidate
            if M == N:
                votes_needed = 0
            else:
                votes_needed = A[sorted_indices[N-M-1]] + 1 - current_votes
            
            if votes_needed <= remaining_votes:
                min_additional_votes[candidate_idx] = max(0, votes_needed)
            else:
                min_additional_votes[candidate_idx] = -1
    
    print(" ".join(map(str, min_additional_votes)))
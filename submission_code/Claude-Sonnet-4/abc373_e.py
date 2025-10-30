def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    total_counted = sum(A)
    remaining_votes = K - total_counted
    
    result = []
    
    for i in range(N):
        # Binary search on additional votes needed
        left, right = 0, remaining_votes + 1
        answer = -1
        
        while left < right:
            mid = (left + right) // 2
            
            if mid > remaining_votes:
                break
                
            # If candidate i gets mid additional votes
            candidate_i_votes = A[i] + mid
            votes_left = remaining_votes - mid
            
            # Find maximum number of candidates that can beat candidate i
            # in the worst case distribution
            other_candidates = []
            for j in range(N):
                if j != i:
                    other_candidates.append(A[j])
            
            # Sort in descending order to prioritize giving votes to strongest candidates
            other_candidates.sort(reverse=True)
            
            # Greedily distribute remaining votes to maximize candidates beating i
            temp_votes = votes_left
            beating_count = 0
            
            for curr_votes in other_candidates:
                if curr_votes > candidate_i_votes:
                    beating_count += 1
                else:
                    # Try to give this candidate enough votes to beat candidate i
                    needed = candidate_i_votes + 1 - curr_votes
                    if needed <= temp_votes:
                        temp_votes -= needed
                        beating_count += 1
            
            # Check if candidate i can win (fewer than M candidates beat them)
            if beating_count < M:
                answer = mid
                right = mid
            else:
                left = mid + 1
        
        result.append(answer)
    
    print(*result)

solve()
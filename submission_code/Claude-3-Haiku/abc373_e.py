from typing import List

def min_additional_votes(N: int, M: int, K: int, A: List[int]) -> List[int]:
    total_votes = sum(A)
    remaining_votes = K - total_votes
    
    result = []
    for i in range(N):
        votes_needed = 0
        candidates_with_more_votes = 0
        for j in range(N):
            if A[j] > A[i]:
                candidates_with_more_votes += 1
        if candidates_with_more_votes < M:
            result.append(0)
        else:
            votes_needed = M - candidates_with_more_votes
            if votes_needed <= remaining_votes:
                result.append(votes_needed)
            else:
                result.append(-1)
    return result
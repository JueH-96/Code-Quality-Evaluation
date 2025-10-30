def can_guarantee_victory(i, X, N, M, K, A):
    # Candidate i will have A[i] + X votes
    target_votes = A[i] + X
    remaining_votes = K - sum(A) - X
    
    if remaining_votes < 0:
        return False
    
    # Count how many candidates can potentially have more votes than candidate i
    # We need to distribute remaining_votes to maximize this count
    
    # For each other candidate j, calculate how many votes they need to surpass target_votes
    needed_votes = []
    for j in range(N):
        if j != i:
            need = max(0, target_votes + 1 - A[j])
            needed_votes.append(need)
    
    # Sort by votes needed (ascending)
    needed_votes.sort()
    
    # Greedily give votes to candidates who need the least
    candidates_with_more = 0
    votes_used = 0
    
    for need in needed_votes:
        if votes_used + need <= remaining_votes:
            votes_used += need
            candidates_with_more += 1
    
    # Candidate i wins if fewer than M candidates have more votes
    return candidates_with_more < M

def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    total_votes_counted = sum(A)
    remaining_votes = K - total_votes_counted
    
    results = []
    
    for i in range(N):
        # Binary search for minimum X
        left, right = 0, remaining_votes
        answer = -1
        
        while left <= right:
            mid = (left + right) // 2
            if can_guarantee_victory(i, mid, N, M, K, A):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        
        results.append(answer)
    
    print(' '.join(map(str, results)))

solve()
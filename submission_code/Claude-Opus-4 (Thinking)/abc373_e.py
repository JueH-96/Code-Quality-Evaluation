# YOUR CODE HERE
def can_win(i, X, N, M, K, A):
    # Candidate i gets X additional votes
    votes_i = A[i] + X
    remaining_votes = K - sum(A) - X
    
    if remaining_votes < 0:
        return False
    
    # Count candidates who can beat candidate i
    candidates_beating_i = 0
    votes_needed = []
    
    for j in range(N):
        if j == i:
            continue
        if A[j] > votes_i:
            candidates_beating_i += 1
        else:
            votes_needed.append(votes_i + 1 - A[j])
    
    # If already too many candidates beat i, return False
    if candidates_beating_i >= M:
        return False
    
    # Sort by votes needed (ascending)
    votes_needed.sort()
    
    # Greedily give votes to candidates
    for needed in votes_needed:
        if candidates_beating_i >= M:
            break
        if remaining_votes >= needed:
            remaining_votes -= needed
            candidates_beating_i += 1
        else:
            break
    
    # Candidate i wins if fewer than M candidates beat them
    return candidates_beating_i < M

N, M, K = map(int, input().split())
A = list(map(int, input().split()))

total_votes_cast = sum(A)
total_remaining = K - total_votes_cast

result = []

for i in range(N):
    # Binary search for the minimum X
    left, right = 0, total_remaining
    answer = -1
    
    while left <= right:
        mid = (left + right) // 2
        if can_win(i, mid, N, M, K, A):
            answer = mid
            right = mid - 1
        else:
            left = mid + 1
    
    result.append(answer)

print(' '.join(map(str, result)))
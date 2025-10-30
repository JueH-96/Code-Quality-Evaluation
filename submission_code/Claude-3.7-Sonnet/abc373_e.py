N, M, K = map(int, input().split())
A = list(map(int, input().split()))

total_votes_so_far = sum(A)
remaining_votes = K - total_votes_so_far
results = []

for i in range(N):
    # Binary search on the number of additional votes for candidate i
    left, right = 0, remaining_votes
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        
        # Total votes for candidate i after getting mid additional votes
        total_votes_i = A[i] + mid
        
        # Count candidates who already have more votes than candidate i
        already_above = sum(1 for j in range(N) if j != i and A[j] > total_votes_i)
        
        # Calculate how many more votes are needed for each candidate to surpass candidate i
        votes_needed = []
        for j in range(N):
            if j == i or A[j] > total_votes_i:
                continue
            votes_needed.append(total_votes_i + 1 - A[j])  # Votes needed to surpass candidate i
        
        # Sort by votes needed in ascending order
        votes_needed.sort()
        
        # Distribute the remaining votes to maximize the number of candidates who can be pushed above candidate i
        available_votes = remaining_votes - mid
        can_push_above = 0
        
        for needed in votes_needed:
            if needed <= available_votes:
                can_push_above += 1
                available_votes -= needed
            else:
                break
        
        total_above = already_above + can_push_above
        
        if total_above < M:
            result = mid
            right = mid - 1
        else:
            left = mid + 1
    
    results.append(result)

print(" ".join(map(str, results)))
def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    remaining = K - sum(A)
    result = []
    
    for i in range(N):
        # If already elected (less than M candidates have more votes)
        more_votes = sum(1 for j in range(N) if A[j] > A[i])
        if more_votes < M:
            result.append(0)
            continue
            
        # Binary search for minimum additional votes needed
        left = 0
        right = remaining + 1
        ans = -1
        
        while left <= right:
            mid = (left + right) // 2
            if mid > remaining:
                left = mid + 1
                continue
                
            # Check if getting mid additional votes guarantees victory
            curr_votes = A[i] + mid
            possible_better = 0
            
            # For each other candidate
            for j in range(N):
                if i == j:
                    continue
                # Maximum votes they can get
                max_other = A[j] + (remaining - mid)
                if max_other >= curr_votes:
                    possible_better += 1
                    
            # If less than M others can get more votes, this works
            if possible_better < M:
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
                
        result.append(ans)
        
    print(*result)

solve()
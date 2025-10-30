N, M, K = map(int, input().split())
A = list(map(int, input().split()))

total_counted = sum(A)
remaining = K - total_counted

results = []
for i in range(N):
    others = [A[j] for j in range(N) if j != i]
    
    def can_guarantee_win(X):
        if X > remaining:
            return False
        
        target = A[i] + X
        left_votes = remaining - X
        
        votes_needed = []
        surpassing = 0
        
        for votes in others:
            if votes > target:
                surpassing += 1
            else:
                need = target + 1 - votes
                votes_needed.append(need)
        
        votes_needed.sort()
        used = 0
        
        for need in votes_needed:
            if used + need <= left_votes:
                used += need
                surpassing += 1
            else:
                break
        
        return surpassing < M
    
    left, right = 0, remaining
    ans = -1
    
    while left <= right:
        mid = (left + right) // 2
        if can_guarantee_win(mid):
            ans = mid
            right = mid - 1
        else:
            left = mid + 1
    
    results.append(ans)

print(' '.join(map(str, results)))
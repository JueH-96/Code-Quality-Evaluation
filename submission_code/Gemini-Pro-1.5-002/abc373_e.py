# YOUR CODE HERE
def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    remaining_votes = k - sum(a)
    
    results = []
    for i in range(n):
        target_votes = a[i]
        
        other_candidates = []
        for j in range(n):
            if i != j:
                other_candidates.append(a[j])
        
        
        def check(x):
            
            my_votes = target_votes + x
            
            
            count = 0
            for j in range(n):
                if i != j:
                    if a[j] > my_votes:
                        count += 1
                    
            
            
            temp_remaining = remaining_votes - x
            
            
            sorted_others = sorted(other_candidates, reverse=True)
            
            
            
            
            
            temp_count = 0
            for j in range(len(sorted_others)):
                if sorted_others[j] > my_votes:
                    temp_count += 1
            
            
            
            
            
            
            return temp_count < m
        
        
        
        low = 0
        high = remaining_votes
        ans = -1
        
        while low <= high:
            mid = (low + high) // 2
            
            if check(mid):
                ans = mid
                high = mid - 1
            else:
                low = mid + 1
        
        results.append(ans)
    
    print(*results)

solve()
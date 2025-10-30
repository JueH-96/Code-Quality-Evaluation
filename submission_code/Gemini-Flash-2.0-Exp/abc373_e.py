def solve():
    n, m, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    total_votes_counted = sum(a)
    remaining_votes = k - total_votes_counted
    
    results = []
    for i in range(n):
        needed_votes = 0
        possible = False
        
        for x in range(remaining_votes + 2):
            
            
            
            
            votes_i = a[i] + x
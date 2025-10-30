# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_sum = sum(a)
    
    ans = float('inf')
    
    for target in range(min(a), max(a) + 2):
        
        current_ops = 0
        
        
        for x in a:
            if x > target:
                current_ops += x - target
            
        
        if total_sum >= target * n and total_sum <= (target + 1) * n:
            ans = min(ans, current_ops)
            
    print(ans)

solve()
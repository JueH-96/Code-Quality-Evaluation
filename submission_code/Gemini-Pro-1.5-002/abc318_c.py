# YOUR CODE HERE
def solve():
    n, d, p = map(int, input().split())
    f = list(map(int, input().split()))
    f.sort(reverse=True)
    
    ans = 0
    for i in range(n):
        ans += f[i]
    
    min_cost = ans
    
    for num_batches in range( (n + d - 1) // d + 1):
        cost = num_batches * p
        passes = num_batches * d
        
        remaining_cost = 0
        for i in range(passes, n):
            remaining_cost += f[i]
        
        min_cost = min(min_cost, cost + remaining_cost)
        
    print(min_cost)

solve()
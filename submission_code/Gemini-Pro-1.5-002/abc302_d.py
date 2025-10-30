# YOUR CODE HERE
def solve():
    n, m, d = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    a.sort()
    b.sort()
    
    max_sum = -1
    
    import bisect
    
    for x in a:
        left = bisect.bisect_left(b, x - d)
        right = bisect.bisect_right(b, x + d)
        
        if left < right:
            max_sum = max(max_sum, x + b[right-1])
            
    print(max_sum)

solve()
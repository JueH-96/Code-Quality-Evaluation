def solve():
    n = int(input())
    x = list(map(int, input().split()))
    q = int(input())
    
    total_moves = 0
    
    for _ in range(q):
        t, g = map(int, input().split())
        t -= 1 
        
        diff = g - x[t]
        total_moves += abs(diff)
        x[t] = g
        
        
        sorted_x = sorted(x)
        x = sorted_x
        
    print(total_moves)

solve()
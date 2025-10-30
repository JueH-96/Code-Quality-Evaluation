# YOUR CODE HERE
def solve():
    N = int(input())
    pairs = [list(map(int, input().split())) for _ in range(N)]
    
    total_min = sum(pair[0] for pair in pairs)
    total_max = sum(pair[1] for pair in pairs)
    
    if total_min > 0 or total_max < 0:
        print("No")
        return
    
    result = []
    target = 0
    
    for L, R in pairs:
        if target - L <= total_max:
            X = max(L, target - total_max)
        else:
            X = min(R, target - total_min)
        
        result.append(X)
        target -= X
        total_min -= L
        total_max -= R
    
    print("Yes")
    print(*result)

solve()
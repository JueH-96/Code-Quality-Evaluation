def solve():
    n = int(input())
    balls = []
    for i in range(n):
        x, y = map(int, input().split())
        balls.append((x, y, i))
    
    MOD = 998244353
    
    # Check if ball i can eliminate ball j when ball i is chosen
    def can_eliminate(i, j):
        xi, yi = balls[i][0], balls[i][1]
        xj, yj = balls[j][0], balls[j][1]
        return (xj < xi and yj < yi) or (xj > xi and yj > yi)
    
    # Check if a subset is valid (no ball can eliminate another)
    def is_valid_subset(subset):
        for i in subset:
            for j in subset:
                if i != j and can_eliminate(i, j):
                    return False
        return True
    
    # Count all valid subsets
    count = 0
    for mask in range(1, 1 << n):  # All non-empty subsets
        subset = []
        for i in range(n):
            if mask & (1 << i):
                subset.append(i)
        
        if is_valid_subset(subset):
            count = (count + 1) % MOD
    
    print(count)

solve()
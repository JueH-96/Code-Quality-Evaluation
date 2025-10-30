def solve():
    n, c = map(int, input().split())
    t = list(map(int, input().split()))
    
    candies = 0
    last_candy_time = -float('inf')
    
    for i in range(n):
        if i == 0 or t[i] - last_candy_time >= c:
            candies += 1
            last_candy_time = t[i]
            
    print(candies)

solve()
def solve():
    n = int(input())
    
    total_weight = 0
    max_weight = 0
    
    for _ in range(n - 1):
        a, b, c = map(int, input().split())
        total_weight += c
        max_weight = max(max_weight, c)
    
    result = 2 * total_weight - max_weight
    print(result)

solve()
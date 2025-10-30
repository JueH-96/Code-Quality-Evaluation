def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    max_val = max(a)
    second_max = -1
    
    for x in a:
        if x != max_val:
            second_max = max(second_max, x)
            
    print(second_max)

solve()
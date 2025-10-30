def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    largest = max(a)
    
    second_largest = -1
    for x in a:
        if x != largest:
            second_largest = max(second_largest, x)
            
    print(second_largest)

solve()
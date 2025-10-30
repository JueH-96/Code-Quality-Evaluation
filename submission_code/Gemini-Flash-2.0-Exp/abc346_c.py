def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    present = set(a)
    
    missing_sum = 0
    for i in range(1, k + 1):
        if i not in present:
            missing_sum += i
            
    print(missing_sum)

solve()
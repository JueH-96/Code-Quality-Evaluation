def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    present = set(a)
    total_sum = 0
    
    for i in range(1, k + 1):
        if i not in present:
            total_sum += i
            
    print(total_sum)

solve()
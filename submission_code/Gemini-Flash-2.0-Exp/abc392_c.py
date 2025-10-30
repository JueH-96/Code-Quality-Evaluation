def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = list(map(int, input().split()))
    
    result = []
    for i in range(n):
        staring_at = p[i]
        staring_at_index = staring_at - 1
        result.append(q[staring_at_index])
    
    print(*result)

solve()
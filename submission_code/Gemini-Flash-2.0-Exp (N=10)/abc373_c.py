def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    max_a = max(a)
    max_b = max(b)
    
    print(max_a + max_b)

solve()
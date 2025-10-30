def solve():
    n = int(input())
    p = list(map(int, input().split()))
    q = int(input())
    
    for _ in range(q):
        a, b = map(int, input().split())
        
        pos_a = -1
        pos_b = -1
        
        for i in range(n):
            if p[i] == a:
                pos_a = i
            if p[i] == b:
                pos_b = i
        
        if pos_a < pos_b:
            print(a)
        else:
            print(b)

solve()
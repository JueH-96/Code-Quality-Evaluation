def solve():
    n = int(input())
    h = list(map(int, input().split()))
    
    first_height = h[0]
    
    for i in range(1, n):
        if h[i] > first_height:
            print(i + 1)
            return
    
    print("-1")

solve()
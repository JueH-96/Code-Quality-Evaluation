def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    total_moves = sum(a)
    
    if n % 2 == 0:
        if total_moves % 2 == 0:
            print("Snuke")
        else:
            print("Fennec")
    else:
        if total_moves % 2 == 0:
            print("Snuke")
        else:
            print("Fennec")

solve()
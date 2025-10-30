def solve():
    n, q = map(int, input().split())
    treatments = list(map(int, input().split()))
    
    teeth = [True] * n
    
    for t in treatments:
        if teeth[t-1]:
            teeth[t-1] = False
        else:
            teeth[t-1] = True
            
    print(sum(teeth))

solve()
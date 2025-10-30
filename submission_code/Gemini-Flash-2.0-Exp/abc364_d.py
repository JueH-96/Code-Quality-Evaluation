def solve():
    n, q = map(int, input().split())
    a = list(map(int, input().split()))
    
    for _ in range(q):
        b, k = map(int, input().split())
        
        distances = []
        for ai in a:
            distances.append(abs(ai - b))
        
        distances.sort()
        
        print(distances[k-1])

solve()
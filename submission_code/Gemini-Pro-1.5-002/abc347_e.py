# YOUR CODE HERE
def solve():
    n, q = map(int, input().split())
    queries = list(map(int, input().split()))
    
    a = [0] * n
    s = set()
    
    for x in queries:
        if x in s:
            s.remove(x)
        else:
            s.insert(x)
        
        for j in range(1, n + 1):
            if j in s:
                a[j-1] += len(s)
                
    print(*a)

solve()
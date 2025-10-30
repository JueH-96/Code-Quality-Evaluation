# YOUR CODE HERE
def solve():
    n, x, y = map(int, input().split())
    p = []
    t = []
    for _ in range(n - 1):
        pi, ti = map(int, input().split())
        p.append(pi)
        t.append(ti)
    
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(int(input()))
    
    for qi in queries:
        current_time = qi + x
        for i in range(n - 1):
            current_time = (current_time + p[i] - 1) // p[i] * p[i] + t[i]
        print(current_time + y)

solve()
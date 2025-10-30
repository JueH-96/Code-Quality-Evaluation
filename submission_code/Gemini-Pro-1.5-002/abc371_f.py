# YOUR CODE HERE
def solve():
    n = int(input())
    x = list(map(int, input().split()))
    q = int(input())
    queries = []
    for _ in range(q):
        queries.append(list(map(int, input().split())))

    ans = 0
    for i in range(q):
        t, g = queries[i]
        t -= 1
        
        diff = g - x[t]
        ans += abs(diff)
        x[t] = g
        
        
    print(ans)

solve()
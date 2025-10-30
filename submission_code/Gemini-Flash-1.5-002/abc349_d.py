# YOUR CODE HERE
def solve():
    l, r = map(int, input().split())
    
    ans = []
    cur = l
    while cur < r:
        nxt = 1
        while nxt <= r - cur:
            nxt *= 2
        nxt //= 2
        ans.append((cur, cur + nxt))
        cur += nxt
    
    print(len(ans))
    for a, b in ans:
        print(a, b)

solve()
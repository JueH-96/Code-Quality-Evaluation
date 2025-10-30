# YOUR CODE HERE
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())
    
    for _ in range(q):
        l, r = map(int, input().split())
        total_sleep = 0
        for i in range(1, n, 2):
            sleep_start = a[i-1]
            sleep_end = a[i]
            
            start = max(l, sleep_start)
            end = min(r, sleep_end)
            
            if start < end:
                total_sleep += end - start
        print(total_sleep)

solve()
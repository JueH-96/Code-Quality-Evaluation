def solve():
    n, q = map(int, input().split())
    r = list(map(int, input().split()))
    r.sort()
    
    for _ in range(q):
        x = int(input())
        count = 0
        total_reindeer = 0
        for i in range(n):
            if total_reindeer + r[i] <= x:
                total_reindeer += r[i]
                count += 1
            else:
                break
        print(count)

solve()
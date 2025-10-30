def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    count = 0
    for s in range(1, n + 1):
        for t in range(1, n + 1):
            if s != t:
                dist = 0
                curr = s
                while curr != t:
                    dist += a[curr - 1]
                    curr = (curr % n) + 1
                if dist % m == 0:
                    count += 1
    print(count)

solve()
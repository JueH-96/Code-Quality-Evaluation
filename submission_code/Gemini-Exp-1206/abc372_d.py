def solve():
    n = int(input())
    h = list(map(int, input().split()))
    
    results = []
    for i in range(n):
        count = 0
        for j in range(i + 1, n):
            valid = True
            for k in range(i + 1, j):
                if h[k] > h[j]:
                    valid = False
                    break
            if valid:
                count += 1
        results.append(count)
    
    print(*results)

solve()
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    count = 0
    used = [False] * n
    
    for i in range(n):
        if used[i]:
            continue
        
        for j in range(i + 1, n):
            if not used[j] and a[i] * 2 <= a[j]:
                count += 1
                used[i] = True
                used[j] = True
                break
    
    print(count)

solve()
def solve():
    n = int(input())
    a = list(map(int, input().split()))

    front = -1
    for i in range(n):
        if a[i] == -1:
            front = i + 1
            break
    
    result = [front]
    current = front
    
    while len(result) < n:
        for i in range(n):
            if a[i] == current:
                result.append(i + 1)
                current = i + 1
                break
    
    print(*result)

solve()
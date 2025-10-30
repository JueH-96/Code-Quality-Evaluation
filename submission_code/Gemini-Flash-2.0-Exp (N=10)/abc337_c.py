def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    front = -1
    for i in range(n):
        if a[i] == -1:
            front = i + 1
            break
    
    result = []
    current = front
    while True:
        result.append(current)
        next_person = -1
        for i in range(n):
            if a[i] == current:
                next_person = i + 1
                break
        if next_person == -1:
            break
        current = next_person
    
    print(*result)

solve()
def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    next_person = {}
    front = -1
    for i in range(n):
        if a[i] == -1:
            front = i + 1
        else:
            next_person[a[i]] = i + 1
    
    result = []
    curr = front
    while curr != None:
        result.append(curr)
        curr = next_person.get(curr)
    
    print(*result)

solve()
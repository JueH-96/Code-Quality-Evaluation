def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())

    for _ in range(q):
        query = list(map(int, input().split()))
        
        if query[0] == 1:
            x = query[1]
            y = query[2]
            
            index_x = a.index(x)
            a.insert(index_x + 1, y)
        else:
            x = query[1]
            a.remove(x)

    print(*a)

solve()
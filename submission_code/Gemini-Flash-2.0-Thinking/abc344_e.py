def solve():
    n = int(input())
    a = list(map(int, input().split()))
    q = int(input())

    next_map = {}
    prev_map = {}

    for i in range(n):
        if i > 0:
            prev_map[a[i]] = a[i-1]
            next_map[a[i-1]] = a[i]

    head = a[0] if n > 0 else None

    for _ in range(q):
        query = input().split()
        query_type = int(query[0])

        if query_type == 1:
            x = int(query[1])
            y = int(query[2])

            next_x = next_map.get(x)
            next_map[x] = y
            prev_map[y] = x
            if next_x is not None:
                prev_map[next_x] = y
                next_map[y] = next_x

        elif query_type == 2:
            x = int(query[1])
            prev_x = prev_map.get(x)
            next_x = next_map.get(x)

            if prev_x is not None:
                next_map[prev_x] = next_x
            else:
                head = next_x

            if next_x is not None:
                prev_map[next_x] = prev_x

            if x in next_map:
                del next_map[x]
            if x in prev_map:
                del prev_map[x]

    result = []
    current = head
    while current is not None:
        result.append(current)
        current = next_map.get(current)

    print(*result)

solve()
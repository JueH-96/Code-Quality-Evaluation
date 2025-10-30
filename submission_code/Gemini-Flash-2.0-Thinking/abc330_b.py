def solve():
    n, l, r = map(int, input().split())
    a = list(map(int, input().split()))

    result = []
    for val in a:
        if val < l:
            result.append(l)
        elif val > r:
            result.append(r)
        else:
            result.append(val)

    print(*result)

# YOUR CODE HERE
solve()
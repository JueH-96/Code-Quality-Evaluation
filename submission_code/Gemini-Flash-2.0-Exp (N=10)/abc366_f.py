def solve():
    n, k = map(int, input().split())
    functions = []
    for _ in range(n):
        functions.append(list(map(int, input().split())))

    import itertools

    max_val = -float('inf')
    for perm in itertools.permutations(range(n), k):
        val = 1
        for i in perm:
            val = functions[i][0] * val + functions[i][1]
        max_val = max(max_val, val)
    print(max_val)

solve()
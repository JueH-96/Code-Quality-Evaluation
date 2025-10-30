def solve():
    n, k = map(int, input().split())
    functions = []
    for _ in range(n):
        functions.append(list(map(int, input().split())))

    def calculate(p):
        x = 1
        for i in reversed(p):
            x = functions[i-1][0] * x + functions[i-1][1]
        return x

    import itertools
    max_val = -1
    for p in itertools.permutations(range(1, n + 1), k):
        max_val = max(max_val, calculate(p))

    print(max_val)

solve()
import itertools

def solve():
    n, k = map(int, input().split())
    functions = []
    for _ in range(n):
        a, b = map(int, input().split())
        functions.append((a, b))

    max_value = 0
    for p in itertools.permutations(range(n), k):
        value = 1
        for i in range(k):
            a, b = functions[p[i]]
            value = a * value + b
        max_value = max(max_value, value)

    print(max_value)

solve()
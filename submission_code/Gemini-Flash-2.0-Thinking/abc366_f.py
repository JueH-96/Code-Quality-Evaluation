import itertools

def solve():
    n, k = map(int, input().split())
    functions = []
    for _ in range(n):
        a, b = map(int, input().split())
        functions.append((a, b))

    max_value = -float('inf')

    for p_tuple in itertools.permutations(range(1, n + 1), k):
        value = 1
        for func_index in reversed(p_tuple):
            a, b = functions[func_index - 1]
            value = a * value + b
        max_value = max(max_value, value)

    print(max_value)

solve()
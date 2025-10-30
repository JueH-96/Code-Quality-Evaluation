def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        a, b = map(int, input().split())
        giants.append((a, b))

    max_head_height = 0
    import itertools
    for perm in itertools.permutations(range(n)):
        current_height = 0
        for i, idx in enumerate(perm):
            a, b = giants[idx]
            if i == 0:
                current_height = b
            else:
                current_height += a
                current_height = current_height - a + b
        max_head_height = max(max_head_height, current_height)
    print(max_head_height)
solve()
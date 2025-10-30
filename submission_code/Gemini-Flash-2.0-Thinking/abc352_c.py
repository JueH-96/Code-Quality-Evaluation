def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        a, b = map(int, input().split())
        giants.append((a, b))

    max_head_height = 0

    import itertools

    for p in itertools.permutations(range(n)):
        current_height = 0
        top_head_height = 0
        for i in range(n):
            giant_index = p[i]
            a = giants[giant_index][0]
            b = giants[giant_index][1]

            if i == 0:
                top_head_height = b
                current_height = a
            else:
                top_head_height = current_height + b
                current_height += a
        max_head_height = max(max_head_height, top_head_height)

    print(max_head_height)

def solve_optimized():
    n = int(input())
    a_values = []
    b_values = []
    for _ in range(n):
        a, b = map(int, input().split())
        a_values.append(a)
        b_values.append(b)

    max_diff = -1
    max_diff_index = -1
    for i in range(n):
        diff = b_values[i] - a_values[i]
        if diff > max_diff:
            max_diff = diff
            max_diff_index = i

    sum_a = sum(a_values)
    max_head_height = sum_a + max_diff

    print(max_head_height)

solve_optimized()
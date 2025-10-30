def solve():
    n, s, m, l = map(int, input().split())

    min_cost = float('inf')

    for num_6 in range(21):
        for num_8 in range(21):
            for num_12 in range(21):
                num_eggs = 6 * num_6 + 8 * num_8 + 12 * num_12
                if num_eggs >= n:
                    cost = num_6 * s + num_8 * m + num_12 * l
                    min_cost = min(min_cost, cost)

    print(min_cost)

solve()
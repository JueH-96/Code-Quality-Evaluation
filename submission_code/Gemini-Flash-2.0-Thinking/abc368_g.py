def solve():
    n = int(input())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    q = int(input())

    def solve_query3(l, r, current_a, current_b):
        possible_values = {0}
        for i in range(l - 1, r):
            new_possible_values = set()
            for val in possible_values:
                add_val = val + current_a[i]
                mul_val = val * current_b[i]
                new_possible_values.add(add_val)
                new_possible_values.add(mul_val)
            possible_values = new_possible_values
        return max(possible_values)

    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            a[query[1] - 1] = query[2]
        elif query[0] == 2:
            b[query[1] - 1] = query[2]
        elif query[0] == 3:
            l, r = query[1], query[2]
            result = solve_query3(l, r, a, b)
            print(result)

solve()
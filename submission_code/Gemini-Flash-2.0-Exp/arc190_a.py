def solve():
    n, m = map(int, input().split())
    operations = []
    for _ in range(m):
        operations.append(tuple(map(int, input().split())))

    best_cost = float('inf')
    best_ops = None

    for i in range(3**m):
        current_ops = []
        temp = i
        for _ in range(m):
            current_ops.append(temp % 3)
            temp //= 3

        x = [0] * n
        cost = 0
        
        for k in range(m):
            l, r = operations[k]
            op = current_ops[k]

            if op == 1:
                cost += 1
                for j in range(n):
                    if l <= j + 1 <= r:
                        x[j] = 1
            elif op == 2:
                cost += 1
                for j in range(n):
                    if not (l <= j + 1 <= r):
                        x[j] = 1
        
        if all(val == 1 for val in x):
            if cost < best_cost:
                best_cost = cost
                best_ops = current_ops

    if best_ops is None:
        print("-1")
    else:
        print(best_cost)
        print(*best_ops)

solve()
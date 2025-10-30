def solve():
    n = int(input())
    lr = []
    for _ in range(n):
        lr.append(list(map(int, input().split())))

    l = [pair[0] for pair in lr]
    r = [pair[1] for pair in lr]

    min_sum = sum(l)
    max_sum = sum(r)

    if 0 < min_sum or 0 > max_sum:
        print("No")
        return

    x = list(l)
    current_sum = sum(x)

    for _ in range(n + 1):  # Iterate at most N times, if no solution, should stabilize
        if current_sum == 0:
            print("Yes")
            print(*x)
            return

        made_change = False
        if current_sum < 0:
            for i in range(n):
                increase = min(-current_sum, r[i] - x[i])
                if increase > 0:
                    x[i] += increase
                    current_sum += increase
                    made_change = True
                    if current_sum == 0:
                        print("Yes")
                        print(*x)
                        return
        elif current_sum > 0:
            for i in range(n):
                decrease = min(current_sum, x[i] - l[i])
                if decrease > 0:
                    x[i] -= decrease
                    current_sum -= decrease
                    made_change = True
                    if current_sum == 0:
                        print("Yes")
                        print(*x)
                        return

        if not made_change and current_sum != 0:
            print("No")
            return

    print("No")

solve()
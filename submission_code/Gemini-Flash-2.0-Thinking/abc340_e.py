def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    for op_b in b:
        held_balls = a[op_b]
        a[op_b] = 0

        if held_balls > 0:
            num_full_passes = held_balls // n
            remainder = held_balls % n

            for i in range(n):
                a[i] += num_full_passes

            for k in range(1, remainder + 1):
                target_box = (op_b + k) % n
                a[target_box] += 1

    print(*a)

solve()
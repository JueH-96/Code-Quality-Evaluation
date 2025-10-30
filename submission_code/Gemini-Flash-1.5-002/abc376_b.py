def solve():
    n, q = map(int, input().split())
    instructions = []
    for _ in range(q):
        instructions.append(input().split())

    left_pos = 1
    right_pos = 2
    total_ops = 0

    for i in range(q):
        hand, target = instructions[i]
        target = int(target)

        ops = 0
        if hand == 'L':
            diff = (target - left_pos) % n
            reverse_diff = (n - diff) % n
            ops = min(diff, reverse_diff)
            left_pos = target
        else:
            diff = (target - right_pos) % n
            reverse_diff = (n - diff) % n
            ops = min(diff, reverse_diff)
            right_pos = target

        total_ops += ops

    print(total_ops)

solve()